/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#pragma once

#if ENABLE(WEBASSEMBLY) && ENABLE(B3_JIT)

#include "WasmTypeDefinition.h"
#include <wtf/Atomics.h>

namespace JSC {
namespace Wasm {

class WasmOpcodeCounter {
    WTF_MAKE_FAST_ALLOCATED;
    using NumberOfRegisteredOpcodes = size_t;
    using CounterSize = size_t;

public:
    static WasmOpcodeCounter& singleton();

    void registerDispatch();

    void increment(ExtSIMDOpType);
    void increment(ExtAtomicOpType);
    void increment(ExtGCOpType);
    void increment(OpType);

    void dump();
    template<typename OpcodeType, typename OpcodeTypeDump, typename IsRegisteredOpcodeFunctor>
    void dump(Atomic<uint64_t>* counter, NumberOfRegisteredOpcodes, CounterSize, const IsRegisteredOpcodeFunctor&, const char* prefix, const char* suffix);

private:
    constexpr static std::pair<NumberOfRegisteredOpcodes, CounterSize> m_extendedSIMDOpcodeInfo = countNumberOfWasmExtendedSIMDOpcodes();
    Atomic<uint64_t> m_extendedSIMDOpcodeCounter[m_extendedSIMDOpcodeInfo.second];

    constexpr static std::pair<NumberOfRegisteredOpcodes, CounterSize> m_extendedAtomicOpcodeInfo = countNumberOfWasmExtendedAtomicOpcodes();
    Atomic<uint64_t> m_extendedAtomicOpcodeCounter[m_extendedAtomicOpcodeInfo.second];

    constexpr static std::pair<NumberOfRegisteredOpcodes, CounterSize> m_GCOpcodeInfo = countNumberOfWasmGCOpcodes();
    Atomic<uint64_t> m_GCOpcodeCounter[m_GCOpcodeInfo.second];

    constexpr static std::pair<NumberOfRegisteredOpcodes, CounterSize> m_baseOpcodeInfo = countNumberOfWasmBaseOpcodes();
    Atomic<uint64_t> m_baseOpcodeCounter[m_baseOpcodeInfo.second];
};

} // namespace JSC
} // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY) && && ENABLE(B3_JIT)
