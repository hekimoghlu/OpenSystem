/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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
#ifndef _UBSAN_MINIMAL_H_
#define _UBSAN_MINIMAL_H_

#if CONFIG_UBSAN_MINIMAL
/*
 * This minimal runtime contains the handlers for checks that are suitable
 * at runtime. To minimize codegen impact, the handlers simply act as a shim
 * to a brk instruction, which gets then inlined by the compiler+LTO.
 * This is similar to UBSAN trapping mode, but guarantees that we can fix
 * and continue by simply stepping to the next instruction during the exception
 * handler.
 *
 * UBSAN Minimal runtime is currently available only for iOS and only for
 * signed overflow checks. It is only used on RELEASE and DEVELOPMENT kernels.
 */

#pragma GCC visibility push(hidden)

/* Trap handler for telemetry */
void ubsan_handle_brk_trap(void *, uint16_t);

/* Setup ubsan minimal runtime */
void ubsan_minimal_init(void);

/*
 * signed-integer-overflow ABI
 */
void __ubsan_handle_divrem_overflow_minimal(void);
void __ubsan_handle_negate_overflow_minimal(void);
void __ubsan_handle_mul_overflow_minimal(void);
void __ubsan_handle_sub_overflow_minimal(void);
void __ubsan_handle_add_overflow_minimal(void);

#pragma GCC visibility pop

#endif /* CONFIG_UBSAN_MINIMAL */
#endif /* _UBSAN_MINIMAL_H_ */
