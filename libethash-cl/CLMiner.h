/// OpenCL miner implementation.
///
/// @file
/// @copyright GNU General Public License

#pragma once

#include <fstream>

#include "ethash_miner_kernel.h"
#include "progpow_miner_kernel.h"

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>
#include <libprogpow/ProgPow.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/lexical_cast.hpp>

#pragma GCC diagnostic push
#if __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#pragma GCC diagnostic ignored "-Wmissing-braces"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS true
#define CL_HPP_ENABLE_EXCEPTIONS true
#define CL_HPP_CL_1_2_DEFAULT_BUILD true
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include "CL/cl2.hpp"
#pragma GCC diagnostic pop

// macOS OpenCL fix:
#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
#endif

#ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
#define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
#endif

#define MAX_SEARCH_RESULTS 4U

// Nvidia GPUs do not play well with blocking queue requests
// and boost host cpu to 100%. These two are used to
// - keep an EMA (Exponential Moving Average) of the search kernel run time
// - keep a ratio to sleep over kernel time.
constexpr double KERNEL_EMA_ALPHA = 0.25;
constexpr double SLEEP_RATIO = 0.95;  // The lower the value the higher CPU usage on Nvidia

namespace dev
{
namespace eth
{
// NOTE: The following struct must match the one defined in
// ethash.cl
typedef struct
{
    cl_uint count;
    cl_uint abort;
    cl_uint rounds;
    struct
    {
        // 64-bit nonce and 8 words for mix hash
        cl_ulong nonce;
        cl_uint mix[8];
    } result[MAX_SEARCH_RESULTS];
} search_results;

struct CLKernelCacheItem
{
    CLKernelCacheItem(ClPlatformTypeEnum _platform, std::string _compute, std::string _name, uint32_t _period,
        unsigned char* _bin, size_t _bin_sz)
      : platform(_platform), compute(_compute), name(_name), period(_period), bin(_bin), bin_sz(_bin_sz)
    {}
    ClPlatformTypeEnum platform;  // OpenCL Platform
    string compute;               // Compute version for Nvidia platform
    string name;                  // Arch name for Amd
    uint32_t period;              // Height of ProgPoW period
    unsigned char* bin;           // Binary/Ptx program
    size_t bin_sz;                // Binary size
};

class CLMiner : public Miner
{
public:
    CLMiner(unsigned _index, CLSettings _settings, DeviceDescriptor& _device);
    ~CLMiner() override = default;

    static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection, std::vector<unsigned>& _platforms);
    static void enumPlatforms();

    static std::vector<CLKernelCacheItem> CLKernelCache;
    static std::mutex cl_kernel_cache_mutex;
    static std::mutex cl_kernel_build_mutex;

    void kick_miner() override;

protected:
    bool initDevice() override;

    bool initEpoch_internal() override;

private:
    void ethash_search() override;
    void progpow_search() override;

    void compileProgPoWKernel(uint32_t _seed, uint32_t _dagelms) override;
    bool loadProgPoWKernel(uint32_t _seed) override;

    void workLoop() override;

    cl::Kernel m_ethash_search_kernel;
    cl::Kernel m_ethash_dag_kernel;
    cl::Kernel m_progpow_search_kernel;

    std::atomic<bool> m_activeKernel = {false};

    cl::Device m_device;
    cl::Context m_context;
    cl::CommandQueue m_queue;
    cl::CommandQueue m_queue_abort;

    long m_ethash_search_kernel_time = 0L;
    long m_progpow_search_kernel_time = 0L;

    cl::Buffer m_dag;
    cl::Buffer m_light;
    cl::Buffer m_header;
    cl::Buffer m_target;
    cl::Buffer m_searchBuffer;

    CLSettings m_settings;

    uint32_t m_zero = 0;
    uint32_t m_one = 1;
    uint32_t m_zerox3[3] = {0, 0, 0};
    uint64_t m_current_target = 0ULL;
};

}  // namespace eth
}  // namespace dev
