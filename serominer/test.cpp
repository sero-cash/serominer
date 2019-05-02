//
// Created by tang zhige on 2019/4/30.
//

#include <libethcore/Farm.h>
#include <libdevcore/CommonData.h>

int main(int argc, char** argv)
{
    const auto& context = progpow::get_global_epoch_context(0);
    auto header = progpow::hash256_from_bytes(dev::fromHex("0x5ffee07b6b16bc6f364c45b84d412138a0b1588edb74e4123e419384435e1691").data());

    auto result=progpow::hash(context,50,header,15017396847274520746);

    cout << "result: " << dev::toHex(result.final_hash.bytes) << "\n";
    cout << "mix: " << dev::toHex(result.mix_hash.bytes) << "\n";

    return 0;
}
