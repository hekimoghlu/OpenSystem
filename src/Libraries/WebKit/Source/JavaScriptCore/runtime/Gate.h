/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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

namespace JSC {

#if ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY) || CPU(ARM64E)

#define JSC_UTILITY_GATES(v) \
    v(jitCagePtr, NoPtrTag) \
    v(tailCallJSEntryPtrTag, NoPtrTag) \
    v(tailCallJSEntrySlowPathPtrTag, NoPtrTag) \
    v(tailCallWithoutUntagJSEntryPtrTag, NoPtrTag) \
    v(loopOSREntry, NoPtrTag) \
    v(entryOSREntry, NoPtrTag) \
    v(wasmOSREntry, NoPtrTag) \
    v(wasmTailCallWasmEntryPtrTag, NoPtrTag) \
    v(wasmIPIntTailCallWasmEntryPtrTag, NoPtrTag) \
    v(exceptionHandler, NoPtrTag) \
    v(returnFromLLInt, NoPtrTag) \
    v(llint_function_for_call_arity_checkUntag, NoPtrTag) \
    v(llint_function_for_call_arity_checkTag, NoPtrTag) \
    v(llint_function_for_construct_arity_checkUntag, NoPtrTag) \
    v(llint_function_for_construct_arity_checkTag, NoPtrTag) \
    v(vmEntryToJavaScript, JSEntryPtrTag) \

#define JSC_JS_GATE_OPCODES(v) \
    v(op_call, JSEntryPtrTag) \
    v(op_call_ignore_result, JSEntryPtrTag) \
    v(op_construct, JSEntryPtrTag) \
    v(op_super_construct, JSEntryPtrTag) \
    v(op_iterator_next, JSEntryPtrTag) \
    v(op_iterator_open, JSEntryPtrTag) \
    v(op_call_varargs, JSEntryPtrTag) \
    v(op_construct_varargs, JSEntryPtrTag) \
    v(op_super_construct_varargs, JSEntryPtrTag) \
    v(op_call_direct_eval_slow, JSEntrySlowPathPtrTag) \

#if ENABLE(WEBASSEMBLY)

#define JSC_WASM_GATE_OPCODES(v) \
    v(wasm_call, WasmEntryPtrTag) \
    v(wasm_call_indirect, WasmEntryPtrTag) \
    v(wasm_call_ref, WasmEntryPtrTag) \
    v(wasm_ipint_call, WasmEntryPtrTag) \

#else
#define JSC_WASM_GATE_OPCODES(v)
#endif

enum class Gate : uint8_t {
#define JSC_DEFINE_GATE_ENUM(gateName, tag) gateName,
#define JSC_DEFINE_OPCODE_GATE_ENUM(gateName, tag) gateName, gateName##_wide16, gateName##_wide32,
    JSC_UTILITY_GATES(JSC_DEFINE_GATE_ENUM)
    JSC_JS_GATE_OPCODES(JSC_DEFINE_OPCODE_GATE_ENUM)
    JSC_WASM_GATE_OPCODES(JSC_DEFINE_OPCODE_GATE_ENUM)
#undef JSC_DEFINE_GATE_ENUM
};

#define JSC_COUNT(gateName, tag) + 1
#define JSC_OPCODE_COUNT(gateName, tag) + 3
static constexpr unsigned numberOfGates = 0 JSC_UTILITY_GATES(JSC_COUNT) JSC_JS_GATE_OPCODES(JSC_OPCODE_COUNT) JSC_WASM_GATE_OPCODES(JSC_OPCODE_COUNT);
#undef JSC_COUNT
#undef JSC_OPCODE_COUNT

#else // not (ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY) || CPU(ARM64E))

// Keep it non-zero to make JSCConfig's array not [0].
static constexpr unsigned numberOfGates = 1;

#endif // ENABLE(JIT_OPERATION_VALIDATION) || ENABLE(JIT_OPERATION_DISASSEMBLY) || CPU(ARM64E)

} // namespace JSC
