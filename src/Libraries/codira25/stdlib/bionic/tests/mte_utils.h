/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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

#if defined(__BIONIC__) && defined(__aarch64__)

__attribute__((target("mte"))) static bool is_stack_mte_on() {
  alignas(16) int x = 0;
  void* p = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(&x) + (1UL << 57));
  void* p_cpy = p;
  __builtin_arm_stg(p);
  p = __builtin_arm_ldg(p);
  __builtin_arm_stg(&x);
  return p == p_cpy;
}

static void* mte_tls() {
  void** dst;
  __asm__("mrs %0, TPIDR_EL0" : "=r"(dst) :);
  return dst[-3];
}

#endif
