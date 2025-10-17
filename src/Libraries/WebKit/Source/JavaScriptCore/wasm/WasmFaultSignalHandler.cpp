/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
#include "WasmFaultSignalHandler.h"

#if ENABLE(WEBASSEMBLY)

#include "ExecutableAllocator.h"
#include "JSWebAssemblyInstance.h"
#include "LLIntData.h"
#include "MachineContext.h"
#include "NativeCalleeRegistry.h"
#include "WasmCallee.h"
#include "WasmCapabilities.h"
#include "WasmContext.h"
#include "WasmExceptionType.h"
#include "WasmMemory.h"
#include "WasmThunks.h"
#include <wtf/CodePtr.h>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/threads/Signals.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace Wasm {

using WTF::CodePtr;

#if CPU(ARM64E) && HAVE(HARDENED_MACH_EXCEPTIONS)
void* presignedTrampoline { nullptr };
#endif

namespace {
namespace WasmFaultSignalHandlerInternal {
static constexpr bool verbose = false;
}
}

static SignalAction trapHandler(Signal signal, SigInfo& sigInfo, PlatformRegisters& context)
{
    RELEASE_ASSERT(signal == Signal::AccessFault);

    auto instructionPointer = MachineContext::instructionPointer(context);
    if (!instructionPointer)
        return SignalAction::NotHandled;
    void* faultingInstruction = instructionPointer->untaggedPtr();
    dataLogLnIf(WasmFaultSignalHandlerInternal::verbose, "starting handler for fault at: ", RawPointer(faultingInstruction));

#if ENABLE(JIT)
    dataLogLnIf(WasmFaultSignalHandlerInternal::verbose, "JIT memory start: ", RawPointer(startOfFixedExecutableMemoryPool()), " end: ", RawPointer(endOfFixedExecutableMemoryPool()));
#endif
    dataLogLnIf(WasmFaultSignalHandlerInternal::verbose, "WasmLLInt memory start: ", RawPointer(untagCodePtr<void*, CFunctionPtrTag>(LLInt::wasmLLIntPCRangeStart)), " end: ", RawPointer(untagCodePtr<void*, CFunctionPtrTag>(LLInt::wasmLLIntPCRangeEnd)));
    // First we need to make sure we are in JIT code or Wasm LLInt code before we can aquire any locks. Otherwise,
    // we might have crashed in code that is already holding one of the locks we want to aquire.
    assertIsNotTagged(faultingInstruction);
    if (isJITPC(faultingInstruction) || LLInt::isWasmLLIntPC(faultingInstruction)) {
        bool faultedInActiveGrowableMemory = false;
        {
            void* faultingAddress = sigInfo.faultingAddress;
            dataLogLnIf(WasmFaultSignalHandlerInternal::verbose, "checking faulting address: ", RawPointer(faultingAddress), " is in an active fast memory");
            faultedInActiveGrowableMemory = Wasm::Memory::addressIsInGrowableOrFastMemory(faultingAddress);
        }
        if (faultedInActiveGrowableMemory) {
            dataLogLnIf(WasmFaultSignalHandlerInternal::verbose, "found active fast memory for faulting address");

            auto didFaultInWasm = [](void* faultingInstruction) -> std::tuple<bool, Wasm::Callee*> {
                if (LLInt::isWasmLLIntPC(faultingInstruction))
                    return { true, nullptr };
                auto& calleeRegistry = NativeCalleeRegistry::singleton();
                Locker locker { calleeRegistry.getLock() };
                for (auto* callee : calleeRegistry.allCallees()) {
                    if (callee->category() != NativeCallee::Category::Wasm)
                        continue;
                    auto* wasmCallee = static_cast<Wasm::Callee*>(callee);
                    auto [start, end] = wasmCallee->range();
                    dataLogLnIf(WasmFaultSignalHandlerInternal::verbose, "function start: ", RawPointer(start), " end: ", RawPointer(end));
                    if (start <= faultingInstruction && faultingInstruction < end) {
                        dataLogLnIf(WasmFaultSignalHandlerInternal::verbose, "found match");
                        return { true, wasmCallee };
                    }
                }
                return { false, nullptr };
            };

            auto [isWasm, callee] = didFaultInWasm(faultingInstruction);
            if (isWasm) {
                auto* instance = jsSecureCast<JSWebAssemblyInstance*>(static_cast<JSCell*>(MachineContext::wasmInstancePointer(context)));
                instance->setFaultPC(faultingInstruction);
#if CPU(ARM64E) && HAVE(HARDENED_MACH_EXCEPTIONS)
                if (g_wtfConfig.signalHandlers.useHardenedHandler) {
                    MachineContext::setInstructionPointer(context, presignedTrampoline);
                    return SignalAction::Handled;
                }
#endif
                MachineContext::setInstructionPointer(context, LLInt::getCodePtr<CFunctionPtrTag>(wasm_throw_from_fault_handler_trampoline_reg_instance));
                return SignalAction::Handled;
            }
        }
    }
    return SignalAction::NotHandled;
}

void activateSignalingMemory()
{
    static std::once_flag once;
    std::call_once(once, [] {
        if (!Wasm::isSupported())
            return;

        if (!Options::useWasmFaultSignalHandler())
            return;

        activateSignalHandlersFor(Signal::AccessFault);
    });
}

void prepareSignalingMemory()
{
    static std::once_flag once;
    std::call_once(once, [] {
        if (!Wasm::isSupported())
            return;

        if (!Options::useWasmFaultSignalHandler())
            return;

#if CPU(ARM64E) && HAVE(HARDENED_MACH_EXCEPTIONS)
        presignedTrampoline = g_wtfConfig.signalHandlers.presignReturnPCForHandler(LLInt::getCodePtr<NoPtrTag>(wasm_throw_from_fault_handler_trampoline_reg_instance));
#endif
        addSignalHandler(Signal::AccessFault, [] (Signal signal, SigInfo& sigInfo, PlatformRegisters& ucontext) {
            return trapHandler(signal, sigInfo, ucontext);
        });
    });
}
    
} } // namespace JSC::Wasm

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(WEBASSEMBLY)
