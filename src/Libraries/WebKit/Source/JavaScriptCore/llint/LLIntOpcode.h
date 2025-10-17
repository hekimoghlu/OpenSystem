/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 29, 2025.
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

#if ENABLE(C_LOOP)

#define FOR_EACH_LLINT_NOJIT_NATIVE_HELPER(macro) \
    FOR_EACH_CLOOP_BYTECODE_HELPER_ID(macro)

#define FOR_EACH_LLINT_NOJIT_RETURN_HELPER(macro) \
    FOR_EACH_CLOOP_RETURN_HELPER_ID(macro)

#else // !ENABLE(C_LOOP)

#define FOR_EACH_LLINT_NOJIT_NATIVE_HELPER(macro) \
    // Nothing to do here. Use the LLInt ASM / JIT impl instead.

#define FOR_EACH_LLINT_NOJIT_RETURN_HELPER(macro) \
    // Nothing to do here. Use the LLInt ASM / JIT impl instead.

#endif // ENABLE(C_LOOP)


#define FOR_EACH_LLINT_NATIVE_HELPER(macro) \
    FOR_EACH_LLINT_NOJIT_NATIVE_HELPER(macro) \
    FOR_EACH_LLINT_NOJIT_RETURN_HELPER(macro)

#define FOR_EACH_LLINT_OPCODE_EXTENSION(macro) \
    FOR_EACH_BYTECODE_HELPER_ID(macro) \
    FOR_EACH_LLINT_NATIVE_HELPER(macro)


#define FOR_EACH_LLINT_OPCODE_WITH_RETURN(macro) \
    macro(op_call) \
    macro(op_call_ignore_result) \
    macro(op_iterator_open) \
    macro(op_iterator_next) \
    macro(op_construct) \
    macro(op_super_construct) \
    macro(op_call_varargs) \
    macro(op_construct_varargs) \
    macro(op_super_construct_varargs) \
    macro(op_get_by_id) \
    macro(op_get_by_id_direct) \
    macro(op_get_length) \
    macro(op_get_by_val) \
    macro(op_put_by_id) \
    macro(op_put_by_val) \
    macro(op_put_by_val_direct) \
    macro(op_in_by_id) \
    macro(op_in_by_val) \
    macro(op_enumerator_get_by_val) \
    macro(op_enumerator_put_by_val) \
    macro(op_enumerator_in_by_val) \

