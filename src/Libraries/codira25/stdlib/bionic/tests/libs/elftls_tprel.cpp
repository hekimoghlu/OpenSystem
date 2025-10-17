/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
// This shared object tests TPREL relocations in the dynamic linker. It's always
// part of static TLS.

// For accesses to these variables, the bfd and lld linkers generate a TPREL
// relocation with no symbol but a non-zero addend.
__attribute__((tls_model("initial-exec"))) static __thread int tls_var_1 = 3;
__attribute__((tls_model("initial-exec"))) static __thread int tls_var_2 = 7;

extern "C" int bump_static_tls_var_1() {
  return ++tls_var_1;
}

extern "C" int bump_static_tls_var_2() {
  return ++tls_var_2;
}

__attribute__((tls_model("initial-exec"), weak)) extern "C" __thread int missing_weak_tls;

extern "C" int* missing_weak_tls_addr() {
  // The dynamic linker should resolve a TPREL relocation to this symbol to 0,
  // which this function adds to the thread pointer.
  return &missing_weak_tls;
}
