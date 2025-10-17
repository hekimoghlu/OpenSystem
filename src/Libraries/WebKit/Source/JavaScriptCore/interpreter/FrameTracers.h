/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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

#include "CatchScope.h"
#include "StackAlignment.h"
#include "VM.h"

namespace JSC {

struct EntryFrame;
class StructureStubInfo;

class SuspendExceptionScope {
public:
    SuspendExceptionScope(VM& vm)
        : m_vm(vm)
        , m_exceptionWasSet(vm.m_exception)
        , m_savedException(vm.m_exception, nullptr)
        , m_savedLastException(vm.m_lastException, nullptr)
    {
        if (m_exceptionWasSet)
            m_vm.traps().clearTrapBit(VMTraps::NeedExceptionHandling);
    }
    ~SuspendExceptionScope()
    {
        if (m_exceptionWasSet)
            m_vm.traps().setTrapBit(VMTraps::NeedExceptionHandling);
    }
private:
    VM& m_vm;
    bool m_exceptionWasSet;
    SetForScope<Exception*> m_savedException;
    SetForScope<Exception*> m_savedLastException;
};

class TopCallFrameSetter {
public:
    TopCallFrameSetter(VM& currentVM, CallFrame* callFrame)
        : vm(currentVM)
        , oldCallFrame(currentVM.topCallFrame)
    {
        currentVM.topCallFrame = callFrame;
    }

    ~TopCallFrameSetter()
    {
        vm.topCallFrame = oldCallFrame;
    }
private:
    VM& vm;
    CallFrame* oldCallFrame;
};

ALWAYS_INLINE static void assertStackPointerIsAligned()
{
#ifndef NDEBUG
#if CPU(X86) && !OS(WINDOWS)
    uintptr_t stackPointer;

    asm("movl %%esp,%0" : "=r"(stackPointer));
    ASSERT(!(stackPointer % stackAlignmentBytes()));
#endif
#endif
}

class SlowPathFrameTracer {
public:
    ALWAYS_INLINE SlowPathFrameTracer(VM& vm, CallFrame* callFrame)
    {
        ASSERT(callFrame);
        ASSERT(reinterpret_cast<void*>(callFrame) < reinterpret_cast<void*>(vm.topEntryFrame));
        assertStackPointerIsAligned();
        vm.topCallFrame = callFrame;
    }
};

class NativeCallFrameTracer {
public:
    ALWAYS_INLINE NativeCallFrameTracer(VM& vm, CallFrame* callFrame)
    {
        ASSERT(callFrame);
        ASSERT(reinterpret_cast<void*>(callFrame) < reinterpret_cast<void*>(vm.topEntryFrame));
        assertStackPointerIsAligned();
        vm.topCallFrame = callFrame;
    }
};

class JITOperationPrologueCallFrameTracer {
public:
    ALWAYS_INLINE JITOperationPrologueCallFrameTracer(VM& vm, CallFrame* callFrame)
#if ASSERT_ENABLED
        : m_vm(vm)
#endif
    {
        UNUSED_PARAM(vm);
        UNUSED_PARAM(callFrame);
        ASSERT(callFrame);
        ASSERT(reinterpret_cast<void*>(callFrame) < reinterpret_cast<void*>(vm.topEntryFrame));
        assertStackPointerIsAligned();
#if USE(BUILTIN_FRAME_ADDRESS)
        // If ASSERT_ENABLED and USE(BUILTIN_FRAME_ADDRESS), prepareCallOperation() will put the frame pointer into vm.topCallFrame.
        // We can ensure here that a call to prepareCallOperation() (or its equivalent) is not missing by comparing vm.topCallFrame to
        // the result of __builtin_frame_address which is passed in as callFrame.
        ASSERT(vm.topCallFrame == callFrame);
        vm.topCallFrame = callFrame;
#endif
    }

#if ASSERT_ENABLED
    ~JITOperationPrologueCallFrameTracer()
    {
        // Fill vm.topCallFrame with invalid value when leaving from JIT operation functions.
        m_vm.topCallFrame = std::bit_cast<CallFrame*>(static_cast<uintptr_t>(0x0badbeef0badbeefULL));
    }

    VM& m_vm;
#endif
};

class ICSlowPathCallFrameTracer {
public:
    inline ICSlowPathCallFrameTracer(VM&, CallFrame*, StructureStubInfo*);

#if ASSERT_ENABLED
    ~ICSlowPathCallFrameTracer()
    {
        // Fill vm.topCallFrame with invalid value when leaving from JIT operation functions.
        m_vm.topCallFrame = std::bit_cast<CallFrame*>(static_cast<uintptr_t>(0x0badbeef0badbeefULL));
    }

    VM& m_vm;
#endif
};


} // namespace JSC
