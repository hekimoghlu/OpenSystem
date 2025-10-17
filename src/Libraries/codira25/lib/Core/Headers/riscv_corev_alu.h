/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#ifndef __RISCV_COREV_ALU_H
#define __RISCV_COREV_ALU_H

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(__riscv_xcvalu)

#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__))

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_abs(long a) {
  return __builtin_abs(a);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_sle(long a, long b) {
  return __builtin_riscv_cv_alu_sle(a, b);
}

static __inline__ long __DEFAULT_FN_ATTRS
__riscv_cv_alu_sleu(unsigned long a, unsigned long b) {
  return __builtin_riscv_cv_alu_sleu(a, b);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_min(long a, long b) {
  return __builtin_elementwise_min(a, b);
}

static __inline__ unsigned long __DEFAULT_FN_ATTRS
__riscv_cv_alu_minu(unsigned long a, unsigned long b) {
  return __builtin_elementwise_min(a, b);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_max(long a, long b) {
  return __builtin_elementwise_max(a, b);
}

static __inline__ unsigned long __DEFAULT_FN_ATTRS
__riscv_cv_alu_maxu(unsigned long a, unsigned long b) {
  return __builtin_elementwise_max(a, b);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_exths(int16_t a) {
  return __builtin_riscv_cv_alu_exths(a);
}

static __inline__ unsigned long __DEFAULT_FN_ATTRS
__riscv_cv_alu_exthz(uint16_t a) {
  return __builtin_riscv_cv_alu_exthz(a);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_extbs(int8_t a) {
  return __builtin_riscv_cv_alu_extbs(a);
}

static __inline__ unsigned long __DEFAULT_FN_ATTRS
__riscv_cv_alu_extbz(uint8_t a) {
  return __builtin_riscv_cv_alu_extbz(a);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_clip(long a,
                                                              unsigned long b) {
  return __builtin_riscv_cv_alu_clip(a, b);
}

static __inline__ unsigned long __DEFAULT_FN_ATTRS
__riscv_cv_alu_clipu(unsigned long a, unsigned long b) {
  return __builtin_riscv_cv_alu_clipu(a, b);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_addN(long a, long b,
                                                              uint8_t shft) {
  return __builtin_riscv_cv_alu_addN(a, b, shft);
}

static __inline__ unsigned long __DEFAULT_FN_ATTRS
__riscv_cv_alu_adduN(unsigned long a, unsigned long b, uint8_t shft) {
  return __builtin_riscv_cv_alu_adduN(a, b, shft);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_addRN(long a, long b,
                                                               uint8_t shft) {
  return __builtin_riscv_cv_alu_addRN(a, b, shft);
}

static __inline__ unsigned long __DEFAULT_FN_ATTRS
__riscv_cv_alu_adduRN(unsigned long a, unsigned long b, uint8_t shft) {
  return __builtin_riscv_cv_alu_adduRN(a, b, shft);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_subN(long a, long b,
                                                              uint8_t shft) {
  return __builtin_riscv_cv_alu_subN(a, b, shft);
}

static __inline__ unsigned long __DEFAULT_FN_ATTRS
__riscv_cv_alu_subuN(unsigned long a, unsigned long b, uint8_t shft) {
  return __builtin_riscv_cv_alu_subuN(a, b, shft);
}

static __inline__ long __DEFAULT_FN_ATTRS __riscv_cv_alu_subRN(long a, long b,
                                                               uint8_t shft) {
  return __builtin_riscv_cv_alu_subRN(a, b, shft);
}

static __inline__ unsigned long __DEFAULT_FN_ATTRS
__riscv_cv_alu_subuRN(unsigned long a, unsigned long b, uint8_t shft) {
  return __builtin_riscv_cv_alu_subuRN(a, b, shft);
}

#endif // defined(__riscv_xcvalu)

#if defined(__cplusplus)
}
#endif

#endif // define __RISCV_COREV_ALU_H
