/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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

#include <wtf/CodePtr.h>
#include <wtf/Function.h>
#include <wtf/Lock.h>
#include <wtf/PlatformRegisters.h>

#if OS(UNIX)

#include <signal.h>
#include <tuple>

#if HAVE(MACH_EXCEPTIONS)
#include <mach/exception_types.h>
#endif
#endif // OS(UNIX)

namespace WTF {

// Note that SIGUSR1 is used in Pthread-based ports except for Darwin to suspend and resume threads.
enum class Signal {
#if OS(UNIX)
    // Usr will always chain to any non-default handler install before us. Since there is no way to know
    // if a signal was intended exclusively for us.
    Usr,

    // These signals will only chain if we don't have a handler that can process them. If there is nothing
    // to chain to we restore the default handler and crash.
#if !OS(DARWIN)
    Abort,
#endif
    FloatingPoint,
    Breakpoint, // Be VERY careful with this, installing a handler for this will break lldb/gdb.
    IllegalInstruction,
    AccessFault, // For posix this is both SIGSEGV and SIGBUS
    NumberOfSignals = AccessFault + 2, // AccessFault is really two signals.
    Unknown = NumberOfSignals
#else // not OS(UNIX)
    FloatingPoint,
    IllegalInstruction,
    AccessFault,
    NumberOfSignals = AccessFault + 1,
    Unknown = NumberOfSignals
#endif
};

enum class SignalAction {
    Handled,
    NotHandled,
    ForceDefault
};

struct SigInfo {
    void* faultingAddress { 0 };
};

using SignalHandler = Function<SignalAction(Signal, SigInfo&, PlatformRegisters&)>;
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
using SignalHandlerMemory = std::aligned_storage<sizeof(SignalHandler), std::alignment_of<SignalHandler>::value>::type;
ALLOW_DEPRECATED_DECLARATIONS_END

struct SignalHandlers {
    static void initialize();
    static void finalize();

    void add(Signal, SignalHandler&&);
    template<typename Func>
    void forEachHandler(Signal, NOESCAPE const Func&) const;

    // We intentionally disallow presigning the return PC on platforms that can't authenticate it so
    // we don't accidentally leave an unfrozen pointer in the heap somewhere.
#if CPU(ARM64E) && HAVE(HARDENED_MACH_EXCEPTIONS)
    void* presignReturnPCForHandler(CodePtr<NoPtrTag>);
#endif

    static constexpr size_t numberOfSignals = static_cast<size_t>(Signal::NumberOfSignals);
    static constexpr size_t maxNumberOfHandlers = 4;

    static_assert(numberOfSignals < std::numeric_limits<uint8_t>::max());

#if HAVE(MACH_EXCEPTIONS)
    mach_port_t exceptionPort;
    exception_mask_t addedExceptions;
    bool useMach;
    // FIXME: It seems like this should be able to be just `HAVE(HARDENED_MACH_EXCEPTIONS)` but X86 had weird issues.
#if CPU(ARM64) && HAVE(HARDENED_MACH_EXCEPTIONS)
    bool useHardenedHandler;
#endif
#else
    static constexpr bool useMach = false;
#endif
    enum class InitState : uint8_t {
        Uninitialized = 0,
        Initializing,
        Finalized,
    };
    InitState initState;

    std::array<uint8_t, numberOfSignals> numberOfHandlers;
    std::array<std::array<SignalHandlerMemory, maxNumberOfHandlers>, numberOfSignals> handlers;

#if OS(UNIX)
    struct sigaction oldActions[numberOfSignals];
#endif
};

// Call this method whenever you want to add a signal handler. This function needs to be called
// before g_wtfConfig is frozen. After the g_wtfConfig is frozen, no additional signal handlers may
// be installed. Any attempt to do so will trigger a crash.
// Note: Your signal handler will be called every time the handler for the desired signal is called.
// Thus it is your responsibility to discern if the signal fired was yours.
// These functions are a one way street i.e. once installed, a signal handler cannot be uninstalled
// and once commited they can't be turned off.
WTF_EXPORT_PRIVATE void addSignalHandler(Signal, SignalHandler&&);
// Note: This function doesn't necessarily activate the signal right away. Signals are
// activated when SignalHandlers::finalize() runs and that signal has been turned on.
// This also only currently does something if using Mach exceptions rather than posix/windows signals.
WTF_EXPORT_PRIVATE void activateSignalHandlersFor(Signal);

#if HAVE(MACH_EXCEPTIONS)
void handleSignalsWithMach();

class Thread;
void registerThreadForMachExceptionHandling(Thread&);
#endif // HAVE(MACH_EXCEPTIONS)

} // namespace WTF

#if HAVE(MACH_EXCEPTIONS)
using WTF::registerThreadForMachExceptionHandling;
using WTF::handleSignalsWithMach;
#endif // HAVE(MACH_EXCEPTIONS)

using WTF::Signal;
using WTF::SigInfo;
using WTF::SignalAction;
using WTF::SignalHandler;
using WTF::addSignalHandler;
using WTF::activateSignalHandlersFor;
