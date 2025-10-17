/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 10, 2024.
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
#include "LLIntExceptions.h"

#include "LLIntCommon.h"
#include "LLIntData.h"
#include "LLIntThunks.h"
#include "WasmContext.h"

#if LLINT_TRACING
#include "CatchScope.h"
#include "Exception.h"
#endif

namespace JSC { namespace LLInt {

JSInstruction* returnToThrow(VM& vm)
{
    UNUSED_PARAM(vm);
#if LLINT_TRACING
    if (UNLIKELY(Options::traceLLIntSlowPath())) {
        auto scope = DECLARE_CATCH_SCOPE(vm);
        dataLog("Throwing exception ", JSValue(scope.exception()), " (returnToThrow).\n");
    }
#endif
    return LLInt::exceptionInstructions();
}

WasmInstruction* wasmReturnToThrow(VM& vm)
{
    UNUSED_PARAM(vm);
#if LLINT_TRACING
    if (UNLIKELY(Options::traceLLIntSlowPath())) {
        auto scope = DECLARE_CATCH_SCOPE(vm);
        dataLog("Throwing exception ", JSValue(scope.exception()), " (returnToThrow).\n");
    }
#endif
    return LLInt::wasmExceptionInstructions();
}

MacroAssemblerCodeRef<ExceptionHandlerPtrTag> callToThrow(VM& vm)
{
    UNUSED_PARAM(vm);
#if LLINT_TRACING
    if (UNLIKELY(Options::traceLLIntSlowPath())) {
        auto scope = DECLARE_CATCH_SCOPE(vm);
        dataLog("Throwing exception ", JSValue(scope.exception()), " (callToThrow).\n");
    }
#endif
#if ENABLE(JIT)
    if (Options::useJIT())
        return LLInt::callToThrowThunk();
#endif
    return LLInt::getCodeRef<ExceptionHandlerPtrTag>(llint_throw_during_call_trampoline);
}

MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleUncaughtException(VM&)
{
#if ENABLE(JIT)
    if (Options::useJIT())
        return handleUncaughtExceptionThunk();
#endif
    return LLInt::getCodeRef<ExceptionHandlerPtrTag>(llint_handle_uncaught_exception);
}

MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleCatch(OpcodeSize size)
{
#if ENABLE(JIT)
    if (Options::useJIT())
        return handleCatchThunk(size);
#endif
    switch (size) {
    case OpcodeSize::Narrow:
        return LLInt::getCodeRef<ExceptionHandlerPtrTag>(op_catch);
    case OpcodeSize::Wide16:
        return LLInt::getWide16CodeRef<ExceptionHandlerPtrTag>(op_catch);
    case OpcodeSize::Wide32:
        return LLInt::getWide32CodeRef<ExceptionHandlerPtrTag>(op_catch);
    }
    RELEASE_ASSERT_NOT_REACHED();
    return {};
}

#if ENABLE(WEBASSEMBLY)
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleWasmCatch(OpcodeSize size)
{
#if ENABLE(JIT)
    if (Options::useJIT())
        return handleWasmCatchThunk(size);
#endif
    WasmOpcodeID opcode = wasm_catch;
    switch (size) {
    case OpcodeSize::Narrow:
        return LLInt::getCodeRef<ExceptionHandlerPtrTag>(opcode);
    case OpcodeSize::Wide16:
        return LLInt::getWide16CodeRef<ExceptionHandlerPtrTag>(opcode);
    case OpcodeSize::Wide32:
        return LLInt::getWide32CodeRef<ExceptionHandlerPtrTag>(opcode);
    }
    RELEASE_ASSERT_NOT_REACHED();
    return { };
}

MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleWasmCatchAll(OpcodeSize size)
{
#if ENABLE(JIT)
    if (Options::useJIT())
        return handleWasmCatchAllThunk(size);
#endif
    WasmOpcodeID opcode = wasm_catch_all;
    switch (size) {
    case OpcodeSize::Narrow:
        return LLInt::getCodeRef<ExceptionHandlerPtrTag>(opcode);
    case OpcodeSize::Wide16:
        return LLInt::getWide16CodeRef<ExceptionHandlerPtrTag>(opcode);
    case OpcodeSize::Wide32:
        return LLInt::getWide32CodeRef<ExceptionHandlerPtrTag>(opcode);
    }
    RELEASE_ASSERT_NOT_REACHED();
    return { };
}

MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleWasmTryTable(WasmOpcodeID opcode, OpcodeSize size)
{
#if ENABLE(JIT)
    if (Options::useJIT())
        return handleWasmTryTableThunk(opcode, size);
#endif
    switch (size) {
    case OpcodeSize::Narrow:
        return LLInt::getCodeRef<ExceptionHandlerPtrTag>(opcode);
    case OpcodeSize::Wide16:
        return LLInt::getWide16CodeRef<ExceptionHandlerPtrTag>(opcode);
    case OpcodeSize::Wide32:
        return LLInt::getWide32CodeRef<ExceptionHandlerPtrTag>(opcode);
    }
    RELEASE_ASSERT_NOT_REACHED();
    return { };
}
#endif // ENABLE(WEBASSEMBLY)

} } // namespace JSC::LLInt
