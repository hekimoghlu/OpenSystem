/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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

#if ENABLE(WEBASSEMBLY)

#include "ExecutionCounter.h"
#include "InstructionStream.h"
#include "Options.h"
#include "VirtualRegister.h"
#include <wtf/FixedVector.h>
#include <wtf/HashMap.h>

namespace JSC { namespace Wasm {

class LLIntTierUpCounter : public BaselineExecutionCounter {
    WTF_MAKE_NONCOPYABLE(LLIntTierUpCounter);

public:
    enum class CompilationStatus : uint8_t {
        NotCompiled = 0,
        Compiling,
        Compiled,
    };

    struct OSREntryData {
        uint32_t loopIndex;
        Vector<VirtualRegister> values;
    };

    LLIntTierUpCounter(UncheckedKeyHashMap<WasmInstructionStream::Offset, OSREntryData>&& osrEntryData)
        : m_osrEntryData(WTFMove(osrEntryData))
    {
        optimizeAfterWarmUp();
        m_compilationStatus.fill(CompilationStatus::NotCompiled);
        m_loopCompilationStatus.fill(CompilationStatus::NotCompiled);
    }

    void optimizeAfterWarmUp()
    {
        setNewThreshold(Options::thresholdForBBQOptimizeAfterWarmUp());
        ASSERT(Options::useWasmLLInt() || checkIfOptimizationThresholdReached());
    }

    bool checkIfOptimizationThresholdReached()
    {
        return checkIfThresholdCrossedAndSet(nullptr);
    }

    void optimizeSoon()
    {
        setNewThreshold(Options::thresholdForBBQOptimizeSoon());
    }

    const OSREntryData& osrEntryDataForLoop(WasmInstructionStream::Offset) const;

    ALWAYS_INLINE CompilationStatus compilationStatus(MemoryMode mode) WTF_REQUIRES_LOCK(m_lock) { return m_compilationStatus[static_cast<MemoryModeType>(mode)]; }
    ALWAYS_INLINE void setCompilationStatus(MemoryMode mode, CompilationStatus status) WTF_REQUIRES_LOCK(m_lock) { m_compilationStatus[static_cast<MemoryModeType>(mode)] = status; }

    ALWAYS_INLINE CompilationStatus loopCompilationStatus(MemoryMode mode) WTF_REQUIRES_LOCK(m_lock) { return m_loopCompilationStatus[static_cast<MemoryModeType>(mode)]; }
    ALWAYS_INLINE void setLoopCompilationStatus(MemoryMode mode, CompilationStatus status) WTF_REQUIRES_LOCK(m_lock) { m_loopCompilationStatus[static_cast<MemoryModeType>(mode)] = status; }

    void resetAndOptimizeSoon(MemoryMode mode);

    Lock m_lock;
private:
    std::array<CompilationStatus, numberOfMemoryModes> m_compilationStatus WTF_GUARDED_BY_LOCK(m_lock);
    std::array<CompilationStatus, numberOfMemoryModes> m_loopCompilationStatus WTF_GUARDED_BY_LOCK(m_lock);
    const UncheckedKeyHashMap<WasmInstructionStream::Offset, OSREntryData> m_osrEntryData;
};

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
