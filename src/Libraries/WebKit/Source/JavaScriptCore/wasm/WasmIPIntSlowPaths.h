/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 15, 2024.
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

#include "CommonSlowPaths.h"
#include "WasmExceptionType.h"
#include "WasmIPIntGenerator.h"
#include "WasmTypeDefinition.h"
#include <wtf/StdLibExtras.h>

namespace JSC {

class JSWebAssemblyInstance;

namespace IPInt {

#define WASM_IPINT_EXTERN_CPP_DECL(name, ...) \
    extern "C" UGPRPair SYSV_ABI ipint_extern_##name(JSWebAssemblyInstance* instance, __VA_ARGS__)

#define WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(name, ...) \
    WASM_IPINT_EXTERN_CPP_DECL(name, __VA_ARGS__) REFERENCED_FROM_ASM WTF_INTERNAL

#define WASM_IPINT_EXTERN_CPP_DECL_1P(name) \
    extern "C" UGPRPair SYSV_ABI ipint_extern_##name(JSWebAssemblyInstance* instance)

#define WASM_IPINT_EXTERN_CPP_HIDDEN_DECL_1P(name) \
    WASM_IPINT_EXTERN_CPP_DECL_1P(name) REFERENCED_FROM_ASM WTF_INTERNAL

#if ENABLE(WEBASSEMBLY_BBQJIT)
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(simd_go_straight_to_bbq, CallFrame* cfr);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(prologue_osr, CallFrame* callFrame);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(loop_osr, CallFrame* callFrame, uint8_t* pc, IPIntLocal* pl);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(epilogue_osr, CallFrame* callFrame);
#endif

WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(retrieve_and_clear_exception, CallFrame*, IPIntStackEntry* stack, IPIntLocal* pl);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(retrieve_clear_and_push_exception, CallFrame*, IPIntStackEntry* stack, IPIntLocal* pl);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(retrieve_clear_and_push_exception_and_arguments, CallFrame*, IPIntStackEntry* stack, IPIntLocal* pl);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(throw_exception, CallFrame*, IPIntStackEntry* arguments, unsigned exceptionIndex);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(rethrow_exception, CallFrame*, IPIntStackEntry* pl, unsigned tryDepth);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(throw_ref, CallFrame* callFrame, EncodedJSValue exnref);

WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(ref_func, unsigned index);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(table_get, unsigned, unsigned);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(table_set, unsigned tableIndex, unsigned index, EncodedJSValue value);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(table_init, IPIntStackEntry* sp, TableInitMetadata* metadata);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(table_fill, IPIntStackEntry* sp, TableFillMetadata* metadata);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(table_grow, IPIntStackEntry* sp, TableGrowMetadata* metadata);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL_1P(current_memory);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(memory_grow, int32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(memory_init, int32_t, IPIntStackEntry*);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(data_drop, int32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(memory_copy, int32_t, int32_t, int32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(memory_fill, int32_t, int32_t, int32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(elem_drop, int32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(table_copy, IPIntStackEntry* sp, TableCopyMetadata* metadata);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(table_size, int32_t);

// Wasm-GC
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(struct_new, Wasm::TypeIndex, IPIntStackEntry* sp);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(struct_new_default, Wasm::TypeIndex);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(struct_get, EncodedJSValue, uint32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(struct_get_s, EncodedJSValue, uint32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(struct_set, EncodedJSValue, uint32_t, IPIntStackEntry*);

WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_new, Wasm::TypeIndex, EncodedJSValue, uint32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_new_default, Wasm::TypeIndex, uint32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_new_fixed, Wasm::TypeIndex, uint32_t, IPIntStackEntry*);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_new_data, IPInt::ArrayNewDataMetadata*, uint32_t, uint32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_new_elem, IPInt::ArrayNewElemMetadata*, uint32_t, uint32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_get, Wasm::TypeIndex, EncodedJSValue, uint32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_get_s, Wasm::TypeIndex, EncodedJSValue, uint32_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_set, Wasm::TypeIndex, IPIntStackEntry*);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_fill, IPIntStackEntry* sp);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_copy, IPIntStackEntry* sp);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_init_data, uint32_t, IPIntStackEntry* sp);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(array_init_elem, uint32_t, IPIntStackEntry* sp);

WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(any_convert_extern, EncodedJSValue);

WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(ref_test, int32_t, bool, EncodedJSValue);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(ref_cast, int32_t, bool, EncodedJSValue);

WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(call_indirect, CallFrame* callFrame, Wasm::FunctionSpaceIndex* functionIndex, CallIndirectMetadata* call);

// We can't use FunctionSpaceIndex here since ARMv7 ABI always passes structs on th stack...
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(prepare_call, unsigned functionSpaceIndex, Register* callee);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(prepare_call_indirect, CallFrame* callFrame, Wasm::FunctionSpaceIndex* functionIndex, CallIndirectMetadata* call);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(prepare_call_ref, CallFrame*, Wasm::TypeIndex, IPIntStackEntry*);

WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(set_global_ref, uint32_t globalIndex, JSValue value);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(get_global_64, unsigned);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(set_global_64, unsigned, uint64_t);

WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(memory_atomic_wait32, uint64_t, uint32_t, uint64_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(memory_atomic_wait64, uint64_t, uint64_t, uint64_t);
WASM_IPINT_EXTERN_CPP_HIDDEN_DECL(memory_atomic_notify, unsigned, unsigned, int32_t);


} } // namespace JSC::IPInt

#endif
