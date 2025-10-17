/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#include <wtf/StdLibExtras.h>

namespace JSC {

class CallFrame;
struct ProtoCallFrame;

template<typename> struct BaseInstruction;
struct WasmOpcodeTraits;
using WasmInstruction = BaseInstruction<WasmOpcodeTraits>;

class JSWebAssemblyInstance;

namespace LLInt {

extern "C" void SYSV_ABI logWasmPrologue(uint64_t i, uint64_t* fp, uint64_t* sp) REFERENCED_FROM_ASM WTF_INTERNAL;

#define WASM_SLOW_PATH_DECL(name) \
    extern "C" UGPRPair SYSV_ABI slow_path_wasm_##name(CallFrame* callFrame, const WasmInstruction* pc, JSWebAssemblyInstance* instance)

#define WASM_SLOW_PATH_HIDDEN_DECL(name) \
    WASM_SLOW_PATH_DECL(name) REFERENCED_FROM_ASM WTF_INTERNAL

#if ENABLE(WEBASSEMBLY_BBQJIT)
WASM_SLOW_PATH_HIDDEN_DECL(prologue_osr);
WASM_SLOW_PATH_HIDDEN_DECL(loop_osr);
WASM_SLOW_PATH_HIDDEN_DECL(epilogue_osr);
WASM_SLOW_PATH_HIDDEN_DECL(simd_go_straight_to_bbq_osr);
#endif

WASM_SLOW_PATH_HIDDEN_DECL(trace);
WASM_SLOW_PATH_HIDDEN_DECL(out_of_line_jump_target);

WASM_SLOW_PATH_HIDDEN_DECL(ref_func);
WASM_SLOW_PATH_HIDDEN_DECL(table_get);
WASM_SLOW_PATH_HIDDEN_DECL(table_set);
WASM_SLOW_PATH_HIDDEN_DECL(table_init);
WASM_SLOW_PATH_HIDDEN_DECL(table_fill);
WASM_SLOW_PATH_HIDDEN_DECL(table_grow);
WASM_SLOW_PATH_HIDDEN_DECL(grow_memory);
WASM_SLOW_PATH_HIDDEN_DECL(memory_init);
WASM_SLOW_PATH_HIDDEN_DECL(call);
WASM_SLOW_PATH_HIDDEN_DECL(call_indirect);

WASM_SLOW_PATH_HIDDEN_DECL(call_ref);
WASM_SLOW_PATH_HIDDEN_DECL(tail_call);
WASM_SLOW_PATH_HIDDEN_DECL(tail_call_indirect);
WASM_SLOW_PATH_HIDDEN_DECL(tail_call_ref);
WASM_SLOW_PATH_HIDDEN_DECL(call_builtin);
WASM_SLOW_PATH_HIDDEN_DECL(set_global_ref);

WASM_SLOW_PATH_HIDDEN_DECL(set_global_ref_portable_binding);
WASM_SLOW_PATH_HIDDEN_DECL(memory_atomic_wait32);
WASM_SLOW_PATH_HIDDEN_DECL(memory_atomic_wait64);
WASM_SLOW_PATH_HIDDEN_DECL(memory_atomic_notify);
WASM_SLOW_PATH_HIDDEN_DECL(throw);
WASM_SLOW_PATH_HIDDEN_DECL(rethrow);
WASM_SLOW_PATH_HIDDEN_DECL(throw_ref);
WASM_SLOW_PATH_HIDDEN_DECL(retrieve_and_clear_exception);
WASM_SLOW_PATH_HIDDEN_DECL(array_new);
WASM_SLOW_PATH_HIDDEN_DECL(array_get);
WASM_SLOW_PATH_HIDDEN_DECL(array_set);
WASM_SLOW_PATH_HIDDEN_DECL(array_fill);
WASM_SLOW_PATH_HIDDEN_DECL(struct_new);
WASM_SLOW_PATH_HIDDEN_DECL(struct_get);
WASM_SLOW_PATH_HIDDEN_DECL(struct_set);

extern "C" NO_RETURN void SYSV_ABI wasm_log_crash(CallFrame*, JSWebAssemblyInstance* instance) REFERENCED_FROM_ASM WTF_INTERNAL;
extern "C" UGPRPair SYSV_ABI slow_path_wasm_throw_exception(CallFrame*, JSWebAssemblyInstance* instance, Wasm::ExceptionType) REFERENCED_FROM_ASM WTF_INTERNAL;
extern "C" UGPRPair SYSV_ABI slow_path_wasm_popcount(const WasmInstruction* pc, uint32_t) REFERENCED_FROM_ASM WTF_INTERNAL;
extern "C" UGPRPair SYSV_ABI slow_path_wasm_popcountll(const WasmInstruction* pc, uint64_t) REFERENCED_FROM_ASM WTF_INTERNAL;

#if USE(JSVALUE32_64)
WASM_SLOW_PATH_HIDDEN_DECL(f32_ceil);
WASM_SLOW_PATH_HIDDEN_DECL(f32_floor);
WASM_SLOW_PATH_HIDDEN_DECL(f32_trunc);
WASM_SLOW_PATH_HIDDEN_DECL(f32_nearest);
WASM_SLOW_PATH_HIDDEN_DECL(f64_ceil);
WASM_SLOW_PATH_HIDDEN_DECL(f64_floor);
WASM_SLOW_PATH_HIDDEN_DECL(f64_trunc);
WASM_SLOW_PATH_HIDDEN_DECL(f64_nearest);
WASM_SLOW_PATH_HIDDEN_DECL(f32_convert_u_i64);
WASM_SLOW_PATH_HIDDEN_DECL(f32_convert_s_i64);
WASM_SLOW_PATH_HIDDEN_DECL(f64_convert_u_i64);
WASM_SLOW_PATH_HIDDEN_DECL(f64_convert_s_i64);
WASM_SLOW_PATH_HIDDEN_DECL(i64_trunc_u_f32);
WASM_SLOW_PATH_HIDDEN_DECL(i64_trunc_s_f32);
WASM_SLOW_PATH_HIDDEN_DECL(i64_trunc_u_f64);
WASM_SLOW_PATH_HIDDEN_DECL(i64_trunc_s_f64);
WASM_SLOW_PATH_HIDDEN_DECL(i64_trunc_sat_f32_u);
WASM_SLOW_PATH_HIDDEN_DECL(i64_trunc_sat_f32_s);
WASM_SLOW_PATH_HIDDEN_DECL(i64_trunc_sat_f64_u);
WASM_SLOW_PATH_HIDDEN_DECL(i64_trunc_sat_f64_s);
#endif

} } // namespace JSC::LLInt

#endif // ENABLE(WEBASSEMBLY)
