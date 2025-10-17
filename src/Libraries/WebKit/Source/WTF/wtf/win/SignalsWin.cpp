/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#include <wtf/threads/Signals.h>

#if OS(WINDOWS)

#include <cstdio>
#include <mutex>
#include <signal.h>
#include <winnt.h>

#include <wtf/Atomics.h>
#include <wtf/DataLog.h>
#include <wtf/MathExtras.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/PlatformRegisters.h>
#include <wtf/ThreadGroup.h>
#include <wtf/Threading.h>
#include <wtf/WTFConfig.h>

namespace WTF {

void SignalHandlers::add(Signal signal, SignalHandler&& handler)
{
    Config::AssertNotFrozenScope assertScope;
    ASSERT(signal < Signal::Unknown);
    RELEASE_ASSERT(initState == SignalHandlers::InitState::Initializing);

    size_t signalIndex = static_cast<size_t>(signal);
    size_t nextFree = numberOfHandlers[signalIndex];
    RELEASE_ASSERT(nextFree < maxNumberOfHandlers);
    SignalHandlerMemory* memory = &handlers[signalIndex][nextFree];
    new (memory) SignalHandler(WTFMove(handler));

    numberOfHandlers[signalIndex]++;
}

template<typename Func>
inline void SignalHandlers::forEachHandler(Signal signal, NOESCAPE const Func& func) const
{
    size_t signalIndex = static_cast<size_t>(signal);
    size_t handlerIndex = numberOfHandlers[signalIndex];
    while (handlerIndex--) {
        auto* memory = const_cast<SignalHandlerMemory*>(&handlers[signalIndex][handlerIndex]);
        const SignalHandler& handler = *std::bit_cast<SignalHandler*>(memory);
        func(handler);
    }
}

inline Signal fromSystemException(DWORD signal)
{
    switch (signal) {
    case EXCEPTION_FLT_DENORMAL_OPERAND:
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
    case EXCEPTION_FLT_INEXACT_RESULT:
    case EXCEPTION_FLT_INVALID_OPERATION:
    case EXCEPTION_FLT_OVERFLOW:
    case EXCEPTION_FLT_STACK_CHECK:
    case EXCEPTION_FLT_UNDERFLOW:
        return Signal::FloatingPoint;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        return Signal::IllegalInstruction;
    case EXCEPTION_ACCESS_VIOLATION:
        return Signal::AccessFault;
    default:
        return Signal::Unknown;
    }
}

LONG WINAPI vectoredHandler(struct _EXCEPTION_POINTERS *exceptionInfo)
{
    Signal signal = fromSystemException(exceptionInfo->ExceptionRecord->ExceptionCode);
    if (signal >= Signal::Unknown)
        return EXCEPTION_CONTINUE_SEARCH;
    SignalHandlers& handlers = g_wtfConfig.signalHandlers;

    SigInfo sigInfo;
    if (signal == Signal::AccessFault) {
        // The second array element specifies the virtual address of the inaccessible data.
        // https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-exception_record
        sigInfo.faultingAddress = reinterpret_cast<void*>(exceptionInfo->ExceptionRecord->ExceptionInformation[1]);
    }

    PlatformRegisters& registers = *(exceptionInfo->ContextRecord);

    long result = EXCEPTION_EXECUTE_HANDLER;
    handlers.forEachHandler(signal, [&] (const SignalHandler& handler) {
        switch (handler(signal, sigInfo, registers)) {
        case SignalAction::Handled:
            result = EXCEPTION_CONTINUE_EXECUTION;
            break;
        default:
            break;
        }
    });

    return result;
}

void addSignalHandler(Signal signal, SignalHandler&& handler)
{
    Config::AssertNotFrozenScope assertScope;
    SignalHandlers& handlers = g_wtfConfig.signalHandlers;
    handlers.add(signal, WTFMove(handler));
}

void activateSignalHandlersFor(Signal signal)
{
    const SignalHandlers& handlers = g_wtfConfig.signalHandlers;
    ASSERT_UNUSED(signal, signal < Signal::Unknown);
    RELEASE_ASSERT(handlers.initState >= SignalHandlers::InitState::Initializing);
}

void SignalHandlers::initialize()
{
    Config::AssertNotFrozenScope assertScope;
    SignalHandlers& handlers = g_wtfConfig.signalHandlers;
    RELEASE_ASSERT(handlers.initState == SignalHandlers::InitState::Uninitialized);
    handlers.initState = SignalHandlers::InitState::Initializing;
}

void SignalHandlers::finalize()
{
    Config::AssertNotFrozenScope assertScope;
    SignalHandlers& handlers = g_wtfConfig.signalHandlers;
    RELEASE_ASSERT(handlers.initState == SignalHandlers::InitState::Initializing);
    handlers.initState = SignalHandlers::InitState::Finalized;

    for (unsigned i = 0; i < numberOfSignals; ++i) {
        if (handlers.numberOfHandlers[i]) {
            AddVectoredExceptionHandler(1, vectoredHandler);
            break;
        }
    }
}

} // namespace WTF

#endif // OS(WINDOWS)
