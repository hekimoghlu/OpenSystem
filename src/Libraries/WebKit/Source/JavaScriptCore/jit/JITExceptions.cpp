/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
#include "JITExceptions.h"

#include "CallFrame.h"
#include "CatchScope.h"
#include "CodeBlock.h"
#include "Interpreter.h"
#include "JSCJSValueInlines.h"
#include "LLIntData.h"
#include "LLIntExceptions.h"
#include "Opcode.h"
#include "ShadowChicken.h"
#include "VMInlines.h"

namespace JSC {

void genericUnwind(VM& vm, CallFrame* callFrame)
{
    auto scope = DECLARE_CATCH_SCOPE(vm);
    CallFrame* topJSCallFrame = vm.topJSCallFrame();
    if (UNLIKELY(Options::breakOnThrow())) {
        CodeBlock* codeBlock = topJSCallFrame->isNativeCalleeFrame() ? nullptr : topJSCallFrame->codeBlock();
        dataLog("In call frame ", RawPointer(topJSCallFrame), " for code block ", codeBlock, "\n");
        WTFBreakpointTrap();
    }
    
    if (auto* shadowChicken = vm.shadowChicken())
        shadowChicken->log(vm, topJSCallFrame, ShadowChicken::Packet::throwPacket());

    Exception* exception = scope.exception();
    RELEASE_ASSERT(exception);
    CatchInfo handler = vm.interpreter.unwind(vm, callFrame, exception); // This may update callFrame.

    void* catchRoutine = nullptr;
    void* dispatchAndCatchRoutine = nullptr;
    JSOrWasmInstruction catchPCForInterpreter = { static_cast<JSInstruction*>(nullptr) };
    uintptr_t catchMetadataPCForInterpreter = 0;
    uint32_t tryDepthForThrow = 0;
    if (handler.m_valid) {
        catchPCForInterpreter = handler.m_catchPCForInterpreter;
        catchMetadataPCForInterpreter = handler.m_catchMetadataPCForInterpreter;
        tryDepthForThrow = handler.m_tryDepthForThrow;
#if ENABLE(JIT)
        catchRoutine = handler.m_nativeCode.taggedPtr();
        if (handler.m_nativeCodeForDispatchAndCatch)
            dispatchAndCatchRoutine = handler.m_nativeCodeForDispatchAndCatch.taggedPtr();
#else
        auto getCatchRoutine = [](const auto* pc) {
            if (pc->isWide32())
                return LLInt::getWide32CodePtr(pc->opcodeID());
            if (pc->isWide16())
                return LLInt::getWide16CodePtr(pc->opcodeID());
            return LLInt::getCodePtr(pc->opcodeID());
        };

        ASSERT_WITH_MESSAGE(!std::holds_alternative<uintptr_t>(catchPCForInterpreter), "IPInt does not support no JIT");
        catchRoutine = std::holds_alternative<const JSInstruction*>(catchPCForInterpreter)
            ? getCatchRoutine(std::get<const JSInstruction*>(catchPCForInterpreter))
            : getCatchRoutine(std::get<const WasmInstruction*>(catchPCForInterpreter));
#endif
    } else
        catchRoutine = LLInt::handleUncaughtException(vm).code().taggedPtr();

    ASSERT(std::bit_cast<uintptr_t>(callFrame) < std::bit_cast<uintptr_t>(vm.topEntryFrame));

    assertIsTaggedWith<ExceptionHandlerPtrTag>(catchRoutine);
    vm.callFrameForCatch = callFrame;
    vm.targetMachinePCForThrow = catchRoutine;
    vm.targetMachinePCAfterCatch = dispatchAndCatchRoutine;
    vm.targetInterpreterPCForThrow = catchPCForInterpreter;
    vm.targetInterpreterMetadataPCForThrow = catchMetadataPCForInterpreter;
    vm.targetTryDepthForThrow = tryDepthForThrow;
    
    RELEASE_ASSERT(catchRoutine);
}

} // namespace JSC
