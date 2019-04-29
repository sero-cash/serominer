/*
 This file is part of ethereum.

 ethminer is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 ethereum is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with ethminer.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Miner.h"

namespace dev
{
namespace eth
{
unsigned Miner::s_dagLoadMode = 0;
unsigned Miner::s_dagLoadIndex = 0;
unsigned Miner::s_minersCount = 0;

FarmFace* FarmFace::m_this = nullptr;

Miner::Miner(std::string const& _name, unsigned _index) : Worker(_name + std::to_string(_index)), m_index(_index)
{
    m_work_latest.header = h256();
}

DeviceDescriptor Miner::getDescriptor()
{
    return m_deviceDescriptor;
}

void Miner::setWork(WorkPackage const& _work)
{
    {
        boost::mutex::scoped_lock l(x_work);
        m_work_latest = _work;
#ifdef _DEVELOPER
        m_workSwitchStart = std::chrono::steady_clock::now();
#endif
    }

    kick_miner();
}

void Miner::stopWorking()
{
    Worker::stopWorking();
    kick_miner();
}

void Miner::kick_miner()
{
    m_new_work.store(true, std::memory_order_relaxed);
    m_new_work_signal.notify_one();
}

void Miner::pause(MinerPauseEnum what)
{
    boost::mutex::scoped_lock l(x_pause);
    m_pauseFlags.set(what);
    kick_miner();
}

bool Miner::paused()
{
    boost::mutex::scoped_lock l(x_pause);
    return m_pauseFlags.any();
}

bool Miner::pauseTest(MinerPauseEnum what)
{
    boost::mutex::scoped_lock l(x_pause);
    return m_pauseFlags.test(what);
}

std::string Miner::pausedString()
{
    boost::mutex::scoped_lock l(x_pause);
    std::string retVar;
    if (m_pauseFlags.any())
    {
        for (int i = 0; i < MinerPauseEnum::Pause_MAX; i++)
        {
            if (m_pauseFlags[(MinerPauseEnum)i])
            {
                if (!retVar.empty())
                    retVar.append("; ");

                if (i == MinerPauseEnum::PauseDueToOverHeating)
                    retVar.append("Overheating");
                else if (i == MinerPauseEnum::PauseDueToAPIRequest)
                    retVar.append("Api request");
                else if (i == MinerPauseEnum::PauseDueToFarmPaused)
                    retVar.append("Farm suspended");
                else if (i == MinerPauseEnum::PauseDueToInsufficientMemory)
                    retVar.append("Insufficient GPU memory");
                else if (i == MinerPauseEnum::PauseDueToInitEpochError)
                    retVar.append("Epoch initialization error");
            }
        }
    }
    return retVar;
}

void Miner::resume(MinerPauseEnum fromwhat)
{
    boost::mutex::scoped_lock l(x_pause);
    m_pauseFlags.reset(fromwhat);
    if (!m_pauseFlags.any())
        kick_miner();
}

float Miner::RetrieveHashRate() noexcept
{
    bool ex = true;
    if (!m_hrLive.compare_exchange_strong(ex, false))
        m_hr.store(0.0f, memory_order_relaxed);

    return m_hr.load(std::memory_order_relaxed);
}

bool Miner::initEpoch()
{
    // When loading of DAG is sequential wait for
    // this instance to become current
    if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
    {
        while (s_dagLoadIndex < m_index)
        {
            boost::system_time const timeout = boost::get_system_time() + boost::posix_time::seconds(3);
            boost::mutex::scoped_lock l(x_work);
            m_dag_loaded_signal.timed_wait(l, timeout);
        }
        if (shouldStop())
            return false;
    }

    // Run the internal initialization
    // specific for miner
    bool result = initEpoch_internal();

    // Advance to next miner or reset to zero for
    // next run if all have processed
    if (s_dagLoadMode == DAG_LOAD_MODE_SEQUENTIAL)
    {
        s_dagLoadIndex = (m_index + 1);
        if (s_minersCount == s_dagLoadIndex)
            s_dagLoadIndex = 0;
        else
            m_dag_loaded_signal.notify_all();
    }

    return result;
}

bool Miner::initEpoch_internal()
{
    // If not overridden in derived class
    this_thread::sleep_for(std::chrono::seconds(5));
    return true;
}

void Miner::minerLoop()
{
    int newEpoch;
    uint32_t newProgPoWPeriod;

    // Don't catch exceptions here !!
    // They will be handled in workLoop implemented in derived class
    while (!shouldStop())
    {
        newEpoch = 0;
        newProgPoWPeriod = 0;

        // Wait for work or 3 seconds (whichever the first)
        if (!m_new_work.load(memory_order_relaxed))
        {
            boost::system_time const timeout = boost::get_system_time() + boost::posix_time::seconds(3);
            boost::mutex::scoped_lock l(x_work);
            m_new_work_signal.timed_wait(l, timeout);
            continue;
        }

        // Got new work
        m_new_work.store(false, memory_order_relaxed);

        if (shouldStop())  // Exit ! Request to terminate
            break;
        if (paused() || !m_work_latest)  // Wait ! Gpu is not ready or there is no work
            continue;

        // Copy latest work into active slot
        {
            boost::mutex::scoped_lock l(x_work);

            // On epoch change for sure we have a period switch
            newEpoch = (m_work_latest.epoch != m_work_active.epoch) ? m_work_latest.epoch : 0;
            if (m_work_latest.algo == "progpow")
            {
                // Check latest period is different from active period
                // This also occurs on epoch change as long as PROGPOW_PERIOD is
                // a divisor of Epoch height (i.e. (30k % PROGPOW_PERIOD) == 0)
                if (m_work_latest.block / PROGPOW_PERIOD != m_work_active.period)
                {
                    newProgPoWPeriod = m_work_latest.block / PROGPOW_PERIOD;
                    m_work_latest.period = int(newProgPoWPeriod);
                }
                else
                {
                    m_work_latest.period = m_work_active.period;

                    // Do get prepared for next period
                    if (uint32_t(m_work_latest.period) >= m_progpow_kernel_latest.load(memory_order_relaxed))
                    {
                        if (((m_work_latest.period + 1) * PROGPOW_PERIOD) % 30000 != 0)
                        {
                            invokeAsyncCompile(uint32_t(m_work_latest.period + 1), false);
                        }
                    }
                }
            }

            // Lower current target so we can be sure it will be set as
            // constant into device
            if (m_work_active.algo != m_work_latest.algo || newEpoch || newProgPoWPeriod)
                m_current_target = 0;

            m_work_active = m_work_latest;
            l.unlock();
        }

        // Epoch change ?
        if (newEpoch)
        {
            // If mining algo is ProgPoW invoke async compilation
            // of kernel while DAG is generating. Epoch context is already loaded
            if (m_work_active.algo == "progpow")
                invokeAsyncCompile(uint32_t(m_work_active.period), false);

            if (!initEpoch())
                break;  // This will simply exit the thread

            // Forces load of new period
            if (m_work_active.algo == "progpow")
                loadProgPoWKernel(newProgPoWPeriod);

            // As DAG generation takes a while we need to
            // ensure we're on latest job, not on the one
            // which triggered the epoch change
            if (m_new_work.load(memory_order_relaxed))
                continue;
        }

        if (m_work_active.algo == "ethash")
        {
            // Start ethash searching
            ethash_search();
        }
        else if (m_work_active.algo == "progpow")
        {
            // If we're switching epoch or period load appropriate
            // kernel from cache
            if (newEpoch || newProgPoWPeriod)
            {
                // If we can't load it it's not in cache
                // Force a sync compilation as last resort
                if (!loadProgPoWKernel(newProgPoWPeriod))
                {
                    invokeAsyncCompile(newProgPoWPeriod, true);
                    if (!loadProgPoWKernel(newProgPoWPeriod))
                    {
                        clog << "Unable to load proper ProgPoW kernel";
                        break;  // Exit the thread
                    }
                }
            }

            // Start progpow searching
            progpow_search();
        }
        else
        {
            throw std::runtime_error("Algo : " + m_work_active.algo + " not yet implemented");
        }
    }

    unloadProgPoWKernel();

    if (m_compilerThread)
    {
        m_compilerThread->join();
        m_compilerThread.reset();
    }
}

void Miner::updateHashRate(uint32_t _hashes, uint64_t _microseconds) noexcept
{
    // Signal we've received an update
    m_hrLive.store(true, memory_order_relaxed);

    // If the sampling interval is too small treat this call as
    // as simple "I'm alive" call.
    // Minimum sampling interval is 1s.
    if (_microseconds < 1000000ULL)
        return;

    float instantHr = float(_hashes * 1.0e6f) / _microseconds;
    m_hr.store(instantHr, memory_order_relaxed);
}

void Miner::invokeAsyncCompile(uint32_t _seed, bool _wait)
{
    if (m_compilerThread)
    {
        if (m_compilerThread->joinable())
            m_compilerThread->join();
    }

    m_progpow_kernel_compile_inprogress.store(true, std::memory_order_relaxed);
    std::string tname = dev::getThreadName();

    m_compilerThread.reset(new std::thread(
        [&](uint32_t _seed, std::string _tname) {
            try
            {
                dev::setThreadName(_tname.c_str());
                compileProgPoWKernel(_seed, uint32_t(m_epochContext.dagSize / ETHASH_MIX_BYTES));
                m_progpow_kernel_latest.store(_seed, memory_order_relaxed);
            }
            catch (const std::runtime_error& _ex)
            {
                cwarn << "Failed to compile ProgPoW kernel : " << _ex.what();
            }
            m_progpow_kernel_compile_inprogress.store(false, std::memory_order_relaxed);
        },
        _seed, tname));

    if (_wait)
    {
        m_compilerThread->join();
        m_compilerThread.reset();
    }
}

}  // namespace eth
}  // namespace dev
