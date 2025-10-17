/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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

#include "IndexingType.h"
#include "JITOperationValidation.h"
#include "JSCJSValue.h"
#include "OperationResult.h"
#include "WasmExceptionType.h"
#include "WasmFormat.h"
#include "WasmOSREntryData.h"
#include "WasmTypeDefinition.h"

namespace JSC {

class JSArray;
class JSWebAssemblyInstance;
class WebAssemblyFunction;

namespace Probe {
class Context;
} // namespace JSC::Probe
namespace Wasm {

class TypeDefinition;
class JSEntrypointCallee;

typedef int64_t EncodedWasmValue;

JSC_DECLARE_JIT_OPERATION(operationJSToWasmEntryWrapperBuildFrame, JSEntrypointCallee*, (void*, CallFrame*, WebAssemblyFunction*));
JSC_DECLARE_JIT_OPERATION(operationJSToWasmEntryWrapperBuildReturnFrame, EncodedJSValue, (void*, CallFrame*));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationGetWasmCalleeStackSize, unsigned, (JSWebAssemblyInstance*, WasmCallableFunction*));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmToJSExitNeedToUnpack, const TypeDefinition*, (WasmOrJSImportableFunctionCallLinkInfo*));
JSC_DECLARE_JIT_OPERATION(operationWasmToJSExitMarshalArguments, bool, (void*, CallFrame*, void*, JSWebAssemblyInstance*));
JSC_DECLARE_JIT_OPERATION(operationWasmToJSExitMarshalReturnValues, void, (void* sp, CallFrame* cfr, JSWebAssemblyInstance*));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmToJSExitIterateResults, bool, (JSWebAssemblyInstance*, const TypeDefinition*, uint64_t* registerResults, uint64_t* calleeFramePointer));

#if ENABLE(WEBASSEMBLY_OMGJIT)
void loadValuesIntoBuffer(Probe::Context&, const StackMap&, uint64_t* buffer, SavedFPWidth);
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmTriggerTierUpNow, void, (CallFrame*, JSWebAssemblyInstance*));
#endif
#if ENABLE(WEBASSEMBLY_OMGJIT) || ENABLE(WEBASSEMBLY_BBQJIT)
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmTriggerOSREntryNow, void, (Probe::Context&));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmLoopOSREnterBBQJIT, void, (Probe::Context&));
#endif
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmUnwind, void*, (JSWebAssemblyInstance*));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationConvertToI64, int64_t, (JSWebAssemblyInstance*, EncodedJSValue));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationConvertToF64, double, (JSWebAssemblyInstance*, EncodedJSValue));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationConvertToI32, int32_t, (JSWebAssemblyInstance*, EncodedJSValue));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationConvertToF32, float, (JSWebAssemblyInstance*, EncodedJSValue));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationConvertToFuncref, EncodedJSValue, (JSWebAssemblyInstance*, const TypeDefinition*, EncodedJSValue));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationConvertToAnyref, EncodedJSValue, (JSWebAssemblyInstance*, const TypeDefinition*, EncodedJSValue));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationConvertToBigInt, EncodedJSValue, (JSWebAssemblyInstance*, EncodedWasmValue));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationIterateResults, bool, (JSWebAssemblyInstance*, const TypeDefinition*, EncodedJSValue, uint64_t*, uint64_t*));
JSC_DECLARE_JIT_OPERATION(operationAllocateResultsArray, JSArray*, (JSWebAssemblyInstance*, const FunctionSignature*, IndexingType, JSValue*));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmWriteBarrierSlowPath, void, (JSCell*, VM*));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationPopcount32, uint32_t, (int32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationPopcount64, uint64_t, (int64_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationGrowMemory, int32_t, (JSWebAssemblyInstance*, int32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmMemoryFill, UCPUStrictInt32, (JSWebAssemblyInstance*, uint32_t dstAddress, uint32_t targetValue, uint32_t count));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmMemoryCopy, UCPUStrictInt32, (JSWebAssemblyInstance*, uint32_t dstAddress, uint32_t srcAddress, uint32_t count));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationGetWasmTableElement, EncodedJSValue, (JSWebAssemblyInstance*, unsigned, int32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationSetWasmTableElement, UCPUStrictInt32, (JSWebAssemblyInstance*, unsigned, uint32_t, EncodedJSValue encValue));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmRefFunc, EncodedJSValue, (JSWebAssemblyInstance*, uint32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmTableInit, UCPUStrictInt32, (JSWebAssemblyInstance*, unsigned elementIndex, unsigned tableIndex, uint32_t dstOffset, uint32_t srcOffset, uint32_t length));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmElemDrop, void, (JSWebAssemblyInstance*, unsigned elementIndex));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmTableGrow, int32_t, (JSWebAssemblyInstance*, unsigned, EncodedJSValue fill, uint32_t delta));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmTableFill, UCPUStrictInt32, (JSWebAssemblyInstance*, unsigned, uint32_t offset, EncodedJSValue fill, uint32_t count));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmTableCopy, UCPUStrictInt32, (JSWebAssemblyInstance*, unsigned dstTableIndex, unsigned srcTableIndex, int32_t dstOffset, int32_t srcOffset, int32_t length));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationGetWasmTableSize, int32_t, (JSWebAssemblyInstance*, unsigned));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationMemoryAtomicWait32, int32_t, (JSWebAssemblyInstance* instance, unsigned base, unsigned offset, int32_t value, int64_t timeout));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationMemoryAtomicWait64, int32_t, (JSWebAssemblyInstance* instance, unsigned base, unsigned offset, int64_t value, int64_t timeout));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationMemoryAtomicNotify, int32_t, (JSWebAssemblyInstance*, unsigned, unsigned, int32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmMemoryInit, UCPUStrictInt32, (JSWebAssemblyInstance*, unsigned dataSegmentIndex, uint32_t dstAddress, uint32_t srcAddress, uint32_t length));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmDataDrop, void, (JSWebAssemblyInstance*, unsigned dataSegmentIndex));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmStructNew, EncodedJSValue, (JSWebAssemblyInstance*, uint32_t, bool, uint64_t*));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmStructNewEmpty, EncodedJSValue, (JSWebAssemblyInstance*, uint32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmStructGet, EncodedJSValue, (EncodedJSValue, uint32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmStructSet, void, (JSWebAssemblyInstance*, EncodedJSValue, uint32_t, EncodedJSValue));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmThrow, void*, (JSWebAssemblyInstance*, unsigned exceptionIndex, uint64_t*));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmRethrow, void*, (JSWebAssemblyInstance*, EncodedJSValue thrownValue));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmToJSException, void*, (JSWebAssemblyInstance*, Wasm::ExceptionType));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationCrashDueToBBQStackOverflow, void, ());
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationCrashDueToOMGStackOverflow, void, ());

struct ThrownExceptionInfo {
    EncodedJSValue thrownValue;
    void* payload;
};
#if USE(JSVALUE64)
static_assert(sizeof(ThrownExceptionInfo) == sizeof(UCPURegister) * 2, "ThrownExceptionInfo should fit in two machine registers");
#endif

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayNew, EncodedJSValue, (JSWebAssemblyInstance* instance, uint32_t typeIndex, uint32_t size, uint64_t value));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayNewVector, EncodedJSValue, (JSWebAssemblyInstance* instance, uint32_t typeIndex, uint32_t size, uint64_t lane0, uint64_t lane1));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayNewData, EncodedJSValue, (JSWebAssemblyInstance* instance, uint32_t typeIndex, uint32_t dataSegmentIndex, uint32_t arraySize, uint32_t offset));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayNewElem, EncodedJSValue, (JSWebAssemblyInstance* instance, uint32_t typeIndex, uint32_t elemSegmentIndex, uint32_t arraySize, uint32_t offset));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayNewEmpty, EncodedJSValue, (JSWebAssemblyInstance* instance, uint32_t typeIndex, uint32_t size));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayGet, EncodedJSValue, (JSWebAssemblyInstance* instance, uint32_t typeIndex, EncodedJSValue encValue, uint32_t index));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArraySet, void, (JSWebAssemblyInstance* instance, uint32_t typeIndex, EncodedJSValue encValue, uint32_t index, EncodedJSValue value));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayFill, UCPUStrictInt32, (JSWebAssemblyInstance*, EncodedJSValue, uint32_t, uint64_t, uint32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayFillVector, UCPUStrictInt32, (JSWebAssemblyInstance*, EncodedJSValue, uint32_t, uint64_t, uint64_t, uint32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayCopy, UCPUStrictInt32, (JSWebAssemblyInstance*, EncodedJSValue, uint32_t, EncodedJSValue, uint32_t, uint32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayInitElem, UCPUStrictInt32, (JSWebAssemblyInstance*, EncodedJSValue, uint32_t, uint32_t, uint32_t, uint32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmArrayInitData, UCPUStrictInt32, (JSWebAssemblyInstance*, EncodedJSValue, uint32_t, uint32_t, uint32_t, uint32_t));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmIsSubRTT, bool, (Wasm::RTT*, Wasm::RTT*));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmAnyConvertExtern, EncodedJSValue, (EncodedJSValue));

JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmRefTest, int32_t, (JSWebAssemblyInstance*, EncodedJSValue, uint32_t, int32_t, bool));
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmRefCast, EncodedJSValue, (JSWebAssemblyInstance*, EncodedJSValue, uint32_t, int32_t));

#if USE(JSVALUE64)
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmRetrieveAndClearExceptionIfCatchable, ThrownExceptionInfo, (JSWebAssemblyInstance*));
#else
/* On 32-bit platforms, we can't return both an EncodedJSValue and the payload
 * together (not enough return registers are available in the ABI and we don't
 * handle stack arguments when calling C functions), so, instead, return the
 * payload and return the thrown value via an out pointer */
JSC_DECLARE_NOEXCEPT_JIT_OPERATION(operationWasmRetrieveAndClearExceptionIfCatchable, void*, (JSWebAssemblyInstance*, EncodedJSValue*));
#endif // USE(JSVALUE64)

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)
