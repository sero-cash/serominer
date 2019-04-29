/*
This file is part of ethminer.

ethminer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ethminer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <libethcore/Farm.h>
#include <ethash/ethash.hpp>

#include "CUDAMiner.h"
#include "nvrtc.h"


using namespace std;
using namespace dev;
using namespace eth;

struct CUDAChannel : public LogChannel
{
    static const char* name() { return EthOrange "cu"; }
    static const int verbosity = 2;
};
#define cudalog clog(CUDAChannel)

std::vector<CUKernelCacheItem> CUDAMiner::CUKernelCache;
std::mutex CUDAMiner::cu_kernel_cache_mutex;
std::mutex CUDAMiner::cu_kernel_build_mutex;

CUDAMiner::CUDAMiner(unsigned _index, CUSettings _settings, DeviceDescriptor& _device)
  : Miner("cuda-", _index),
    m_settings(_settings),
    m_batch_size(_settings.gridSize * _settings.blockSize),
    m_streams_batch_size(_settings.gridSize * _settings.blockSize * CUDA_STREAMS)
{
    m_deviceDescriptor = _device;
}

bool CUDAMiner::initDevice()
{
    cudalog << "Using Pci Id : " << m_deviceDescriptor.uniqueId << " " << m_deviceDescriptor.cuName
            << " (Compute " + m_deviceDescriptor.cuCompute + ") Memory : "
            << dev::getFormattedMemory((double)m_deviceDescriptor.totalMemory) << " Grid size : " << m_settings.gridSize
            << " Block size : " << m_settings.blockSize;

    // Set Hardware Monitor Info
    m_hwmoninfo.deviceType = HwMonitorInfoType::NVIDIA;
    m_hwmoninfo.devicePciId = m_deviceDescriptor.uniqueId;
    m_hwmoninfo.deviceIndex = -1;  // Will be later on mapped by nvml (see Farm() constructor)

    try
    {
        CUDA_SAFE_CALL(cudaSetDevice(m_deviceDescriptor.cuDeviceIndex));
        CUDA_SAFE_CALL(cudaDeviceReset());
        CUDA_SAFE_CALL(cudaSetDeviceFlags(4));  // sync mode
        CUDA_SAFE_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

        // create mining buffers
        for (unsigned i = 0; i < CUDA_STREAMS; ++i)
        {
            CUDA_SAFE_CALL(cudaMallocHost(m_search_results + i, sizeof(search_results)));
            CUDA_SAFE_CALL(cudaStreamCreateWithFlags(m_streams + i, cudaStreamNonBlocking));
        }
    }
    catch (const cuda_runtime_error& ec)
    {
        cudalog << "Could not set CUDA device on Pci Id " << m_deviceDescriptor.uniqueId << " Error : " << ec.what();
        cudalog << "Mining aborted on this device.";
        return false;
    }
    return true;
}

bool CUDAMiner::initEpoch_internal()
{
    // If we get here it means epoch has changed so it's not necessary
    // to check again dag sizes. They're changed for sure
    bool retVar = false;
    m_current_target = 0;
    auto startInit = std::chrono::steady_clock::now();
    size_t RequiredMemory = (m_epochContext.dagSize + m_epochContext.lightSize);

    // Release the pause flag if any
    resume(MinerPauseEnum::PauseDueToInsufficientMemory);
    resume(MinerPauseEnum::PauseDueToInitEpochError);

    try
    {
        // If we have already enough memory allocated, we just have to
        // copy light_cache and regenerate the DAG
        if (m_allocated_memory_dag < m_epochContext.dagSize ||
            m_allocated_memory_light_cache < m_epochContext.lightSize)
        {
            // Clear previously allocated memory space
            if (m_allocated_memory_dag > 0)
            {
                CUDA_SAFE_CALL(cudaFree(m_light));
                CUDA_SAFE_CALL(cudaFree(m_dag));
            }

            // TODO Device reset has unloaded all
            m_progpow_kernel_loaded = false;

            // Check whether the current device has sufficient memory every time we recreate the dag
            if (m_deviceDescriptor.totalMemory < RequiredMemory)
            {
                cudalog << "Epoch " << m_epochContext.epochNumber << " requires "
                        << dev::getFormattedMemory((double)RequiredMemory) << " memory.";
                cudalog << "This device hasn't available. Mining suspended ...";
                pause(MinerPauseEnum::PauseDueToInsufficientMemory);
                return true;  // This will prevent to exit the thread and
                              // Eventually resume mining when changing coin or epoch (NiceHash)
            }

            cudalog << "Generating DAG + Light : " << dev::getFormattedMemory((double)RequiredMemory);

            // create buffer for cache
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&m_light), m_epochContext.lightSize));
            m_allocated_memory_light_cache = m_epochContext.lightSize;
            CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>(&m_dag), m_epochContext.dagSize));
            m_allocated_memory_dag = m_epochContext.dagSize;
        }
        else
        {
            cudalog << "Generating DAG + Light (reusing buffers): " << dev::getFormattedMemory((double)RequiredMemory);
            get_constants(&m_dag, NULL, &m_light, NULL);
        }


        CUDA_SAFE_CALL(cudaMemcpy(reinterpret_cast<void*>(m_light), m_epochContext.lightCache, m_epochContext.lightSize,
            cudaMemcpyHostToDevice));

        set_constants(m_dag, m_epochContext.dagNumItems, m_light,
            m_epochContext.lightNumItems);  // in ethash_cuda_miner_kernel.cu

        ethash_generate_dag(m_epochContext.dagSize, m_settings.gridSize, m_settings.blockSize, m_streams[0]);

        cudalog << "Generated DAG + Light in "
                << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startInit)
                       .count()
                << " ms. " << dev::getFormattedMemory((double)(m_deviceDescriptor.totalMemory - RequiredMemory))
                << " left.";

        m_dag_progpow = reinterpret_cast<hash64_t*>(m_dag);
        retVar = true;
    }
    catch (const cuda_runtime_error& ec)
    {
        cudalog << "Unexpected error " << ec.what() << " on CUDA device " << m_deviceDescriptor.uniqueId;
        cudalog << "Mining suspended ...";
        pause(MinerPauseEnum::PauseDueToInitEpochError);
        retVar = true;
    }

    return retVar;
}

void CUDAMiner::workLoop()
{
    m_active_streams[0] = false;
    m_active_streams[1] = false;

    if (!initDevice())
        return;

    try
    {
        minerLoop();                        // In base class Miner
        CUDA_SAFE_CALL(cudaDeviceReset());  // Reset miner and stop working
    }
    catch (cuda_runtime_error const& _e)
    {
        string _what = "GPU error: ";
        _what.append(_e.what());
        throw std::runtime_error(_what);
    }
}

bool CUDAMiner::loadProgPoWKernel(uint32_t _seed)
{
    unloadProgPoWKernel();

    // Get ptx from cache
    char* ptx;
    const char* loweredName;
    bool found = false;
    {
        // Lookup kernel in cache
        std::lock_guard<std::mutex> cache_mtx(CUDAMiner::cu_kernel_cache_mutex);
        for (const CUKernelCacheItem& item : CUKernelCache)
        {
            if (item.compute == m_deviceDescriptor.cuCompute && item.period == _seed)
            {
                ptx = item.ptx;
                loweredName = item.name.c_str();
                found = true;
                break;
            }
        }
    }

    if (!found)
        return false;

    // Load the generated PTX and get a handle to the kernel.
    char* jitInfo = new char[32 * 1024];
    char* jitErr = new char[32 * 1024];
    CUjit_option jitOpt[] = {CU_JIT_INFO_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_LOG_VERBOSE, CU_JIT_GENERATE_LINE_INFO};
    void* jitOptVal[] = {jitInfo, jitErr, (void*)(32 * 1024), (void*)(32 * 1024), (void*)(1), (void*)(0)};


    CU_SAFE_CALL(cuModuleLoadDataEx(&m_module, ptx, 6, jitOpt, jitOptVal));
    CU_SAFE_CALL(cuModuleGetFunction(&m_kernel, m_module, loweredName));
    m_progpow_kernel_loaded = true;

    delete[] jitInfo;
    delete[] jitErr;

    // Allocate space for header and target constants
    if (!d_pheader)
    {
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_pheader, sizeof(hash32_t)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&d_ptarget, sizeof(uint64_t)));
    }

    return true;
}

void CUDAMiner::compileProgPoWKernel(uint32_t _seed, uint32_t _dagelms)
{
    {
        // Delete from cache older periods
        uint32_t latest = m_progpow_kernel_latest.load(memory_order_relaxed);
        std::lock_guard<std::mutex> cache_mtx(CUDAMiner::cu_kernel_cache_mutex);
        for (size_t i = 0; i < CUDAMiner::CUKernelCache.size(); i++)
        {
            const CUKernelCacheItem& item = CUDAMiner::CUKernelCache.at(i);
            if (item.period + 2 < latest)
            {
                CUDAMiner::CUKernelCache.at(i) = std::move(CUDAMiner::CUKernelCache.back());
                CUDAMiner::CUKernelCache.pop_back();
            }
        }
    }

    std::lock_guard<std::mutex> cache_mtx(CUDAMiner::cu_kernel_build_mutex);
    {
        // See if another thread have compiled the needed kernel already
        std::lock_guard<std::mutex> cache_mtx(CUDAMiner::cu_kernel_cache_mutex);
        for (const CUKernelCacheItem& item : CUDAMiner::CUKernelCache)
            if (item.compute == m_deviceDescriptor.cuCompute && item.period == _seed)
                return;
    }

#ifdef _DEVELOPER
    auto startCompile = std::chrono::steady_clock::now();
    if (g_logOptions & LOG_COMPILE)
    {
        cudalog << "Compiling ProgPoW kernel at period " << _seed;
    }
#endif

    // Getting here means no other thread has compiled this kernel
    // Build it now
    const char* name = "progpow_search";
    std::string source = ProgPow::getKern(_seed, _dagelms, ProgPow::KERNEL_CUDA);
    source += cu_progpow_miner_kernel;

#ifdef _DEVELOPER
    if (g_logOptions & LOG_COMPILE)
    {
        // Save generated source for debug purpouses
        std::string fileName = "kernel-" + m_deviceDescriptor.cuCompute + "-" + to_string(_seed) + ".cu";
        std::string tmpDir;

#ifdef _WIN32
        tmpDir = getenv("TEMP");
        tmpDir.append("\\");
#else
        tmpDir = "/tmp/";
#endif

        std::string tmpFile = tmpDir + fileName;
        cudalog << "Dumping kernel to : " << tmpFile;
        ofstream write;
        write.open(tmpFile);
        write << source;
        write.close();
    }
#endif  // _DEVELOPER

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,  // prog
        source.c_str(),                        // buffer
        NULL,                                  // name
        0,                                     // numHeaders
        NULL,                                  // headers
        NULL));                                // includeNames

    NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, name));
    std::string op_arch =
        "-arch=compute_" + to_string(m_deviceDescriptor.cuComputeMajor) + to_string(m_deviceDescriptor.cuComputeMinor);

    const char* opts[] = {op_arch.c_str(),
        // "-lineinfo",      // For debug only
        "-use_fast_math" /*, "-default-device"*/};
    nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
        2,                                                 // numOptions
        opts);                                             // options

#ifdef _DEVELOPER

    if (g_logOptions & LOG_COMPILE)
    {
        // Obtain compilation log from the program.
        size_t logSize;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        char* log = new char[logSize];
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
        cudalog << "Compile log: " << log;
        delete[] log;
    }

#endif

    NVRTC_SAFE_CALL(compileResult);

    // Obtain PTX from the program.
    size_t ptxSize;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char* ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

    // Find the mangled name
    const char* mangledName;
    NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, name, &mangledName));

    std::string loweredName(mangledName);

    // Destroy the program.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

#ifdef _DEVELOPER
    if (g_logOptions & LOG_COMPILE)
    {
        // Save generated source for debug purpouses
        std::string fileName = "kernel-" + m_deviceDescriptor.cuCompute + "-" + to_string(_seed) + ".cu.ptx";
        std::string tmpDir;

#ifdef _WIN32
        tmpDir = getenv("TEMP");
        tmpDir.append("\\");
#else
        tmpDir = "/tmp/";
#endif

        std::string tmpFile = tmpDir + fileName;
        cudalog << "Dumping ptx to : " << tmpFile;
        ofstream write;
        write.open(tmpFile);
        write << ptx;
        write.close();
    }
#endif  // _DEVELOPER

    // Cache the generated Ptx
    {
        std::lock_guard<std::mutex> cache_mtx(CUDAMiner::cu_kernel_cache_mutex);
        CUKernelCache.emplace_back(m_deviceDescriptor.cuCompute, int(_seed), ptx, loweredName);
    }

#ifdef _DEVELOPER
    if (g_logOptions & LOG_COMPILE)
        cudalog << "Done compiling in "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - startCompile)
                       .count()
                << " ms. ";
#endif
}

void CUDAMiner::unloadProgPoWKernel()
{
    if (!m_progpow_kernel_loaded)
        return;

    // Ensure next ProgPoW target is set on next
    // search
    m_current_target = 0;

    CU_SAFE_CALL(cuModuleUnload(m_module));
    m_progpow_kernel_loaded = false;
}

int CUDAMiner::getNumDevices()
{
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err == cudaSuccess)
        return deviceCount;

    if (err == cudaErrorInsufficientDriver)
    {
        int driverVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        if (driverVersion == 0)
            std::cerr << "CUDA Error : No CUDA driver found" << std::endl;
        else
            std::cerr << "CUDA Error : Insufficient CUDA driver " << std::to_string(driverVersion) << std::endl;
    }
    else
    {
        std::cerr << "CUDA Error : " << cudaGetErrorString(err) << std::endl;
    }

    return 0;
}

void CUDAMiner::enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection)
{
    int numDevices = getNumDevices();

    for (int i = 0; i < numDevices; i++)
    {
        string uniqueId;
        ostringstream s;
        DeviceDescriptor deviceDescriptor;
        cudaDeviceProp props;

        try
        {
            CUDA_SAFE_CALL(cudaGetDeviceProperties(&props, i));
            s << setw(2) << setfill('0') << hex << props.pciBusID << ":" << setw(2) << props.pciDeviceID << ".0";
            uniqueId = s.str();

            if (_DevicesCollection.find(uniqueId) != _DevicesCollection.end())
                deviceDescriptor = _DevicesCollection[uniqueId];
            else
                deviceDescriptor = DeviceDescriptor();

            deviceDescriptor.name = string(props.name);
            deviceDescriptor.cuDetected = true;
            deviceDescriptor.uniqueId = uniqueId;
            deviceDescriptor.type = DeviceTypeEnum::Gpu;
            deviceDescriptor.cuDeviceIndex = i;
            deviceDescriptor.cuDeviceOrdinal = i;
            deviceDescriptor.cuName = string(props.name);
            deviceDescriptor.totalMemory = props.totalGlobalMem;
            deviceDescriptor.cuCompute = (to_string(props.major) + "." + to_string(props.minor));
            deviceDescriptor.cuComputeMajor = props.major;
            deviceDescriptor.cuComputeMinor = props.minor;


            _DevicesCollection[uniqueId] = deviceDescriptor;
        }
        catch (const cuda_runtime_error& _e)
        {
            std::cerr << _e.what() << std::endl;
        }
    }
}

void CUDAMiner::ethash_search()
{
    using namespace std::chrono;

    m_workSearchDuration = 0;
    m_workHashes = 0;

    uint64_t startNonce, target;

    startNonce = m_work_active.startNonce;

    set_header(*reinterpret_cast<hash32_t const*>(m_work_active.header.data()));

    target = (uint64_t)(u64)((u256)m_work_active.boundary >> 192);
    if (m_current_target != target)
    {
        m_current_target = target;
        set_target(m_current_target);
    }


    unsigned int launchIndex, streamIndex;
    launchIndex = 0;
    uint32_t found_count = 0;

    while (true)
    {
        streamIndex = launchIndex % CUDA_STREAMS;
        cudaStream_t stream = m_streams[streamIndex];

        volatile search_results& buffer(*m_search_results[streamIndex]);

        if (launchIndex >= CUDA_STREAMS || !m_new_work.load(memory_order_relaxed))
        {
            if (m_active_streams[streamIndex])
            {
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
                found_count = std::min((unsigned)buffer.count, MAX_SEARCH_RESULTS);
                buffer.count = 0;
                m_active_streams[streamIndex] = false;
                m_workSearchDuration = duration_cast<microseconds>(steady_clock::now() - m_workSearchStart).count();
                m_workHashes += m_batch_size;
            }
        }

        if (!m_new_work.load(memory_order_relaxed))
        {
            if (launchIndex == 0)
            {
                m_workSearchStart = steady_clock::now();
#ifdef _DEVELOPER
                // Optionally log job switch time
                if (g_logOptions & LOG_SWITCH)
                    cudalog << "Switch time: "
                            << std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - m_workSwitchStart)
                                   .count()
                            << " ms.";
#endif
            }

            run_ethash_search(m_settings.gridSize, m_settings.blockSize, stream, &buffer, startNonce);
            m_active_streams[streamIndex] = true;
            startNonce += m_batch_size;
        }

        if (found_count)
        {
            // Extract solution and pass to higer level
            // using io_service as dispatcher
            for (uint32_t i = 0; i < found_count; i++)
            {
                h256 mix;
                uint64_t nonce = buffer.result[i].nonce;
                memcpy(mix.data(), (void*)&buffer.result[i].mix, sizeof(buffer.result[i].mix));
                auto sol = Solution{nonce, mix, m_work_active, std::chrono::steady_clock::now(), m_index};

                cudalog << EthWhite << "Job: " << m_work_active.header.abridged()
                        << " Sol: " << toHex(sol.nonce, HexPrefix::Add) << EthReset;

                Farm::f().submitProof(sol);
            }
        }

        // Update the hash rate
        updateHashRate(m_workHashes, m_workSearchDuration);

        if (!m_active_streams[0] && !m_active_streams[1])
            break;

        launchIndex++;
    }
}

void CUDAMiner::progpow_search()
{
    bool hack_false = false;

    using namespace std::chrono;

    m_workSearchDuration = 0;
    m_workHashes = 0;

    uint64_t startNonce, target;

    startNonce = m_work_active.startNonce;

    hash32_t header = *reinterpret_cast<hash32_t const*>(m_work_active.header.data());
    CUDA_SAFE_CALL(cudaMemcpy(d_pheader, &header, sizeof(hash32_t), cudaMemcpyHostToDevice));

    target = (uint64_t)(u64)((u256)m_work_active.boundary >> 192);
    if (m_current_target != target)
    {
        m_current_target = target;
        CUDA_SAFE_CALL(cudaMemcpy(d_ptarget, &target, sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    unsigned int launchIndex, streamIndex;
    launchIndex = 0;
    uint32_t found_count = 0;

    while (true)
    {
        streamIndex = launchIndex % CUDA_STREAMS;
        cudaStream_t stream = m_streams[streamIndex];

        volatile search_results* buffer = m_search_results[streamIndex];

        if (launchIndex >= CUDA_STREAMS || !m_new_work.load(memory_order_relaxed))
        {
            if (m_active_streams[streamIndex])
            {
                CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
                found_count = std::min((unsigned)buffer->count, MAX_SEARCH_RESULTS);
                buffer->count = 0;
                m_active_streams[streamIndex] = false;
                m_workSearchDuration = duration_cast<microseconds>(steady_clock::now() - m_workSearchStart).count();
                m_workHashes += m_batch_size;
            }
        }

        if (!m_new_work.load(memory_order_relaxed))
        {
            if (launchIndex == 0)
            {
                m_workSearchStart = steady_clock::now();

#ifdef _DEVELOPER
                // Optionally log job switch time
                if (g_logOptions & LOG_SWITCH)
                    cudalog << "Switch time: "
                            << std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - m_workSwitchStart)
                                   .count()
                            << " ms.";
#endif
            }

            // Run the batch for this stream
            uint64_t batchNonce = startNonce;
            void* args[] = {&batchNonce, &d_pheader, &d_ptarget, &m_dag_progpow, &buffer, &hack_false};
            CU_SAFE_CALL(cuLaunchKernel(m_kernel, m_settings.gridSize, 1, 1,  // grid dim
                m_settings.blockSize, 1, 1,                                   // block dim
                0,                                                            // shared mem
                stream,                                                       // stream
                args, 0));                                                    // arguments
            m_active_streams[streamIndex] = true;
            startNonce += m_batch_size;
        }

        // Submit solution while kernel running
        if (found_count)
        {
            // Extract solution and pass to higer level
            // using io_service as dispatcher
            for (uint32_t i = 0; i < found_count; i++)
            {
                h256 mix;
                uint64_t nonce = buffer->result[i].nonce;
                memcpy(mix.data(), (void*)&buffer->result[i].mix, sizeof(buffer->result[i].mix));
                auto sol = Solution{nonce, mix, m_work_active, std::chrono::steady_clock::now(), m_index};

                cudalog << EthWhite << "Job: " << m_work_active.header.abridged()
                        << " Sol: " << toHex(sol.nonce, HexPrefix::Add) << EthReset;

                Farm::f().submitProof(sol);
            }

            found_count = 0;
        }

        // Update the hash rate
        updateHashRate(m_workHashes, m_workSearchDuration);

        if (!m_active_streams[0] && !m_active_streams[1])
            break;

        launchIndex++;
    }
}
