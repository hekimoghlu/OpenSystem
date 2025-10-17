/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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

#include "MacroAssemblerCodeRef.h"
#include "OpcodeSize.h"
#include "VM.h"
#include <wtf/Scope.h>

namespace JSC {

struct ProtoCallFrame;
typedef int64_t EncodedJSValue;

extern "C" {
    EncodedJSValue SYSV_ABI vmEntryToJavaScript(void*, VM*, ProtoCallFrame*);
    EncodedJSValue SYSV_ABI vmEntryToNative(void*, VM*, ProtoCallFrame*);
    EncodedJSValue SYSV_ABI vmEntryCustomGetter(JSGlobalObject*, EncodedJSValue, PropertyName, void*);
    void SYSV_ABI vmEntryCustomSetter(JSGlobalObject*, EncodedJSValue, EncodedJSValue, PropertyName, void*);
    EncodedJSValue SYSV_ABI vmEntryHostFunction(JSGlobalObject*, CallFrame*, void*);

#if CPU(ARM64) && CPU(ADDRESS64) && !ENABLE(C_LOOP)
    EncodedJSValue vmEntryToJavaScriptWith0Arguments(void*, VM*, CodeBlock*, JSObject*, JSValue);
    EncodedJSValue vmEntryToJavaScriptWith1Arguments(void*, VM*, CodeBlock*, JSObject*, JSValue, JSValue);
    EncodedJSValue vmEntryToJavaScriptWith2Arguments(void*, VM*, CodeBlock*, JSObject*, JSValue, JSValue, JSValue);
    EncodedJSValue vmEntryToJavaScriptWith3Arguments(void*, VM*, CodeBlock*, JSObject*, JSValue, JSValue, JSValue, JSValue);
#endif
}

#if CPU(ARM64E) && !ENABLE(C_LOOP)
extern "C" {
    void jitCagePtrGateAfter(void);
    void vmEntryToJavaScriptGateAfter(void);

    // WebCore calls these from SelectorCompiler, so they are not hidden like normal LLInt symbols.
    JS_EXPORT_PRIVATE unsigned SYSV_ABI vmEntryToCSSJIT(uintptr_t, uintptr_t, uintptr_t, const void* codePtr);
    JS_EXPORT_PRIVATE void SYSV_ABI vmEntryToCSSJITAfter(void);

    void llint_function_for_call_arity_checkUntagGateAfter(void);
    void llint_function_for_call_arity_checkTagGateAfter(void);
    void llint_function_for_construct_arity_checkUntagGateAfter(void);
    void llint_function_for_construct_arity_checkTagGateAfter(void);
}
#endif

inline EncodedJSValue vmEntryToWasm(void* code, VM* vm, ProtoCallFrame* frame)
{
    auto clobberizeValidator = makeScopeExit([&] {
        vm->didEnterVM = true;
    });
    code = retagCodePtr<WasmEntryPtrTag, JSEntryPtrTag>(code);
    return vmEntryToJavaScript(code, vm, frame);
}

namespace LLInt {

MacroAssemblerCodeRef<JSEntryPtrTag> functionForCallEntryThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> functionForConstructEntryThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> functionForCallArityCheckThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> functionForConstructArityCheckThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> evalEntryThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> programEntryThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> moduleProgramEntryThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> defaultCallThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> getHostCallReturnValueThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> genericReturnPointThunk(OpcodeSize);
MacroAssemblerCodeRef<JSEntryPtrTag> fuzzerReturnEarlyFromLoopHintThunk();
#if ENABLE(JIT)
MacroAssemblerCodeRef<JITThunkPtrTag> arityFixupThunk();
#endif

MacroAssemblerCodeRef<ExceptionHandlerPtrTag> callToThrowThunk();
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleUncaughtExceptionThunk();
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleCatchThunk(OpcodeSize);

#if ENABLE(WEBASSEMBLY)
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleWasmCatchThunk(OpcodeSize);
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleWasmCatchAllThunk(OpcodeSize);
MacroAssemblerCodeRef<ExceptionHandlerPtrTag> handleWasmTryTableThunk(WasmOpcodeID, OpcodeSize);
#endif

#if ENABLE(JIT_CAGE)
MacroAssemblerCodeRef<NativeToJITGatePtrTag> jitCagePtrThunk();
#endif

#if CPU(ARM64E)
MacroAssemblerCodeRef<NativeToJITGatePtrTag> createJSGateThunk(void*, PtrTag, const char*);
MacroAssemblerCodeRef<NativeToJITGatePtrTag> createWasmGateThunk(void*, PtrTag, const char*);
MacroAssemblerCodeRef<NativeToJITGatePtrTag> createTailCallGate(PtrTag, bool);
MacroAssemblerCodeRef<NativeToJITGatePtrTag> createWasmTailCallGate(PtrTag);
MacroAssemblerCodeRef<NativeToJITGatePtrTag> loopOSREntryGateThunk();
MacroAssemblerCodeRef<NativeToJITGatePtrTag> entryOSREntryGateThunk();
MacroAssemblerCodeRef<NativeToJITGatePtrTag> wasmOSREntryGateThunk();
MacroAssemblerCodeRef<NativeToJITGatePtrTag> exceptionHandlerGateThunk();
MacroAssemblerCodeRef<NativeToJITGatePtrTag> returnFromLLIntGateThunk();
MacroAssemblerCodeRef<NativeToJITGatePtrTag> untagGateThunk(void*);
MacroAssemblerCodeRef<NativeToJITGatePtrTag> tagGateThunk(void*);
#endif

MacroAssemblerCodeRef<JSEntryPtrTag> normalOSRExitTrampolineThunk();
#if ENABLE(DFG_JIT)
MacroAssemblerCodeRef<JSEntryPtrTag> checkpointOSRExitTrampolineThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> checkpointOSRExitFromInlinedCallTrampolineThunk();
MacroAssemblerCodeRef<JSEntryPtrTag> returnLocationThunk(OpcodeID, OpcodeSize);
#endif

#if ENABLE(WEBASSEMBLY)
MacroAssemblerCodeRef<JITThunkPtrTag> wasmFunctionEntryThunk();
MacroAssemblerCodeRef<JITThunkPtrTag> wasmFunctionEntryThunkSIMD();
MacroAssemblerCodeRef<JITThunkPtrTag> inPlaceInterpreterEntryThunk();
MacroAssemblerCodeRef<JITThunkPtrTag> inPlaceInterpreterSIMDEntryThunk();
#endif // ENABLE(WEBASSEMBLY)

} } // namespace JSC::LLInt
