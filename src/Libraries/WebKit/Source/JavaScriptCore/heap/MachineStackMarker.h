/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

#include "RegisterState.h"
#include <wtf/Lock.h>
#include <wtf/ScopedLambda.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadGroup.h>

namespace JSC {

class CodeBlockSet;
class ConservativeRoots;
class Heap;
class JITStubRoutineSet;

struct CurrentThreadState {
    void* stackOrigin { nullptr };
    void* stackTop { nullptr };
    RegisterState* registerState { nullptr };
};
    
class MachineThreads {
    WTF_MAKE_TZONE_ALLOCATED(MachineThreads);
    WTF_MAKE_NONCOPYABLE(MachineThreads);
public:
    MachineThreads();

    void gatherConservativeRoots(ConservativeRoots&, JITStubRoutineSet&, CodeBlockSet&, CurrentThreadState*, Thread*);

    // Only needs to be called by clients that can use the same heap from multiple threads.
    bool addCurrentThread() { return m_threadGroup->addCurrentThread() == ThreadGroupAddResult::NewlyAdded; }

    WordLock& getLock() { return m_threadGroup->getLock(); }
    const ListHashSet<Ref<Thread>>& threads(const AbstractLocker& locker) const { return m_threadGroup->threads(locker); }

private:
    void gatherFromCurrentThread(ConservativeRoots&, JITStubRoutineSet&, CodeBlockSet&, CurrentThreadState&);

    void tryCopyOtherThreadStack(const ThreadSuspendLocker&, Thread&, void*, size_t capacity, size_t*);
    bool tryCopyOtherThreadStacks(const AbstractLocker&, void*, size_t capacity, size_t*, Thread&);

    std::shared_ptr<ThreadGroup> m_threadGroup;
};

#define DECLARE_AND_COMPUTE_CURRENT_THREAD_STATE(stateName) \
    CurrentThreadState stateName; \
    stateName.stackTop = &stateName; \
    stateName.stackOrigin = Thread::current().stack().origin(); \
    ALLOCATE_AND_GET_REGISTER_STATE(stateName ## _registerState); \
    stateName.registerState = &stateName ## _registerState

// The return value is meaningless. We just use it to suppress tail call optimization.
int callWithCurrentThreadState(const ScopedLambda<void(CurrentThreadState&)>&);

} // namespace JSC

