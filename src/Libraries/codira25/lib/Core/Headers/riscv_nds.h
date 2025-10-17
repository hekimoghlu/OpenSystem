/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#ifndef __RISCV_NDS_H
#define __RISCV_NDS_H

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__riscv_xandesbfhcvt)

#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))

static __inline__ float __DEFAULT_FN_ATTRS __riscv_nds_fcvt_s_bf16(__bf16 bf) {
  return __builtin_riscv_nds_fcvt_s_bf16(bf);
}

static __inline__ __bf16 __DEFAULT_FN_ATTRS __riscv_nds_fcvt_bf16_s(float sf) {
  return __builtin_riscv_nds_fcvt_bf16_s(sf);
}

#endif // defined(__riscv_xandesbfhcvt)

#if defined(__cplusplus)
}
#endif

#endif // define __RISCV_NDS_H
