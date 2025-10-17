/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "config.h"
#include "WasmOpcodeCounter.h"

#if ENABLE(WEBASSEMBLY) && ENABLE(B3_JIT)
#include "WasmTypeDefinition.h"
#include <wtf/Atomics.h>
#include <wtf/DataLog.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Vector.h>

#if PLATFORM(COCOA)
#include <notify.h>
#include <unistd.h>
#endif

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {
namespace Wasm {

WasmOpcodeCounter& WasmOpcodeCounter::singleton()
{
    static LazyNeverDestroyed<WasmOpcodeCounter> counter;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        counter.construct();
    });
    return counter;
}


template<typename OpcodeType, typename OpcodeTypeDump, typename IsRegisteredOpcodeFunctor>
void WasmOpcodeCounter::dump(Atomic<uint64_t>* counter, NumberOfRegisteredOpcodes numberOfRegisteredOpcode, CounterSize counterSize, const IsRegisteredOpcodeFunctor& isRegisteredOpcodeFunctor, const char* prefix, const char* suffix)
{
    struct Pair {
        OpcodeType opcode;
        uint64_t count;
    };

    Vector<Pair> vector;
    uint64_t usedOpcode = 0;
    for (size_t i = 0; i < counterSize; i++) {
        if (!isRegisteredOpcodeFunctor((OpcodeType)i))
            continue;

        uint64_t count = counter[i].loadFullyFenced();
        if (count)
            usedOpcode++;
        vector.append({ (OpcodeType)i, count });
    }

    std::sort(vector.begin(), vector.end(), [](Pair& a, Pair& b) {
        return b.count < a.count;
    });

    int pid = 0;
#if PLATFORM(COCOA)
    pid = getpid();
#endif
    float coverage = usedOpcode * 1.0 / numberOfRegisteredOpcode * 100;
    dataLogF("%s<%d> %s use coverage %.2f%%.\n", prefix, pid, suffix, coverage);
    for (Pair& pair : vector)
        dataLogLn(prefix, "<", pid, ">    ", OpcodeTypeDump(pair.opcode), ": ", pair.count);
}

void WasmOpcodeCounter::dump()
{
    dump<ExtSIMDOpType, ExtSIMDOpTypeDump>(m_extendedSIMDOpcodeCounter, m_extendedSIMDOpcodeInfo.first, m_extendedSIMDOpcodeInfo.second, isRegisteredWasmExtendedSIMDOpcode, "<WASM.EXT.SIMD.OP.STAT>", "wasm extended SIMD opcode");

    dump<ExtAtomicOpType, ExtAtomicOpTypeDump>(m_extendedAtomicOpcodeCounter, m_extendedAtomicOpcodeInfo.first, m_extendedAtomicOpcodeInfo.second, isRegisteredExtenedAtomicOpcode, "<WASM.EXT.ATOMIC.OP.STAT>", "wasm extended atomic opcode");

    dump<ExtGCOpType, ExtGCOpTypeDump>(m_GCOpcodeCounter, m_GCOpcodeInfo.first, m_GCOpcodeInfo.second, isRegisteredGCOpcode, "<WASM.GC.OP.STAT>", "wasm GC opcode");

    dump<OpType, OpTypeDump>(m_baseOpcodeCounter, m_baseOpcodeInfo.first, m_baseOpcodeInfo.second, isRegisteredBaseOpcode, "<WASM.BASE.OP.STAT>", "wasm base opcode");
}

void WasmOpcodeCounter::registerDispatch()
{
#if PLATFORM(COCOA)
    static std::once_flag registerFlag;
    std::call_once(registerFlag, [&]() {
        int pid = getpid();
        const char* key = "com.apple.WebKit.wasm.op.stat";
        dataLogF("<WASM.OP.STAT><%d> Registering callback for wasm opcode statistics.\n", pid);
        dataLogF("<WASM.OP.STAT><%d> Use `notifyutil -v -p %s` to dump statistics.\n", pid, key);

        int token;
        notify_register_dispatch(key, &token, dispatch_get_main_queue(), ^(int) {
            WasmOpcodeCounter::singleton().dump();
        });
    });
#endif
}

void WasmOpcodeCounter::increment(ExtSIMDOpType op)
{
    registerDispatch();
    m_extendedSIMDOpcodeCounter[(uint8_t)op].exchangeAdd(1);
}

void WasmOpcodeCounter::increment(ExtAtomicOpType op)
{
    registerDispatch();
    m_extendedAtomicOpcodeCounter[(uint8_t)op].exchangeAdd(1);
}

void WasmOpcodeCounter::increment(ExtGCOpType op)
{
    registerDispatch();
    m_GCOpcodeCounter[(uint8_t)op].exchangeAdd(1);
}

void WasmOpcodeCounter::increment(OpType op)
{
    registerDispatch();
    m_baseOpcodeCounter[(uint8_t)op].exchangeAdd(1);
}

} // namespace JSC
} // namespace JSC::Wasm

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(WEBASSEMBLY) && && ENABLE(B3_JIT)
