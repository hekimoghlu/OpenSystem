/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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

#include "Debugger.h"
#include "EntryFrame.h"
#include "FuzzerAgent.h"
#include "ProfilerDatabase.h"
#include "SideDataRepository.h"
#include "VM.h"
#include "Watchdog.h"

namespace JSC {

inline ActiveScratchBufferScope::ActiveScratchBufferScope(ScratchBuffer* buffer, size_t activeScratchBufferSizeInJSValues)
    : m_scratchBuffer(buffer)
{
    // Tell GC mark phase how much of the scratch buffer is active during the call operation this scope is used in.
    if (m_scratchBuffer)
        m_scratchBuffer->setActiveLength(activeScratchBufferSizeInJSValues * sizeof(EncodedJSValue));
}

inline ActiveScratchBufferScope::~ActiveScratchBufferScope()
{
    // Tell the GC that we're not using the scratch buffer anymore.
    if (m_scratchBuffer)
        m_scratchBuffer->setActiveLength(0);
}

bool VM::ensureStackCapacityFor(Register* newTopOfStack)
{
#if !ENABLE(C_LOOP)
    return newTopOfStack >= m_softStackLimit;
#else
    return ensureStackCapacityForCLoop(newTopOfStack);
#endif
    
}

bool VM::isSafeToRecurseSoft() const
{
    bool safe = isSafeToRecurse(m_softStackLimit);
#if ENABLE(C_LOOP)
    safe = safe && isSafeToRecurseSoftCLoop();
#endif
    return safe;
}

template<typename Func>
void VM::logEvent(CodeBlock* codeBlock, const char* summary, const Func& func)
{
    if (LIKELY(!m_perBytecodeProfiler))
        return;
    
    m_perBytecodeProfiler->logEvent(codeBlock, summary, func());
}

inline CallFrame* VM::topJSCallFrame() const
{
    CallFrame* frame = topCallFrame;
    if (UNLIKELY(!frame))
        return frame;
    if (LIKELY(!frame->isNativeCalleeFrame() && !frame->isPartiallyInitializedFrame()))
        return frame;
    EntryFrame* entryFrame = topEntryFrame;
    do {
        frame = frame->callerFrame(entryFrame);
        ASSERT(!frame || !frame->isPartiallyInitializedFrame());
    } while (frame && frame->isNativeCalleeFrame());
    return frame;
}

inline void VM::setFuzzerAgent(std::unique_ptr<FuzzerAgent>&& fuzzerAgent)
{
    RELEASE_ASSERT_WITH_MESSAGE(!m_fuzzerAgent, "Only one FuzzerAgent can be specified at a time.");
    m_fuzzerAgent = WTFMove(fuzzerAgent);
}

template<typename Func>
inline void VM::forEachDebugger(const Func& callback)
{
    if (LIKELY(m_debuggers.isEmpty()))
        return;

    for (auto* debugger = m_debuggers.head(); debugger; debugger = debugger->next())
        callback(*debugger);
}

template<typename Type, typename Functor>
Type& VM::ensureSideData(void* key, const Functor& functor)
{
    m_hasSideData = true;
    return sideDataRepository().ensure<Type>(this, key, functor);
}

} // namespace JSC
