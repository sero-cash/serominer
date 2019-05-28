/*
This file is part of serominer.

serominer is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

serominer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with serominer.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <libdevcore/Worker.h>
#include <libethcore/EthashAux.h>
#include <libethcore/Miner.h>

#include <functional>
#include <chrono>

namespace dev
{
namespace eth
{
class CPUMiner : public Miner
{
public:
    CPUMiner(unsigned _index, CPSettings _settings, DeviceDescriptor& _device);
    ~CPUMiner() override = default;

    static unsigned getNumDevices();
    static void enumDevices(std::map<string, DeviceDescriptor>& _DevicesCollection);

protected:
    bool initDevice() override;

private:
    void progpow_search() override;
    void compileProgPoWKernel(uint32_t _seed, uint32_t _dagelms) override;
    bool loadProgPoWKernel(uint32_t _seed) override;

    void workLoop() override;

    CPSettings m_settings;
    std::chrono::steady_clock::time_point start_time;
    uint32_t hash_count;
};


}  // namespace eth
}  // namespace dev
