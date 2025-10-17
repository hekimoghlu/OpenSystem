/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
// This shared object test library is dlopen'ed by the main test executable.

// Export a large TLS variable from a solib for testing dladdr and dlsym. The
// TLS symbol's value will appear to overlap almost everything else in the
// shared object, but dladdr must not return it.
__thread char large_tls_var[4 * 1024 * 1024];

extern "C" char* get_large_tls_var_addr() {
  return large_tls_var;
}

// For testing dladdr, return an address that's part of the solib's .bss
// section, but does not have an entry in the dynsym table and whose
// solib-relative address appears to overlap with the large TLS variable.
extern "C" void* get_local_addr() {
  static char buf[1024];
  return &buf[512];
}

// This variable comes from libtest_elftls_shared_var.so, which is part of
// static TLS. Verify that a GD-model access can access the variable.
//
// Accessing the static TLS variable from an solib prevents the static linker
// from relaxing the GD access to IE and lets us test that __tls_get_addr and
// the tlsdesc resolver handle a static TLS variable.
extern "C" __thread int elftls_shared_var;

extern "C" int bump_shared_var() {
  return ++elftls_shared_var;
}

// The static linker denotes the current module by omitting the symbol from
// the DTPMOD/TLSDESC relocations.
static __thread int local_var_1 = 15;
static __thread int local_var_2 = 25;

extern "C" int bump_local_vars() {
  return ++local_var_1 + ++local_var_2;
}

extern "C" int get_local_var1() {
  return local_var_1;
}

extern "C" int* get_local_var1_addr() {
  return &local_var_1;
}

extern "C" int get_local_var2() {
  return local_var_2;
}

__attribute__((weak)) extern "C" __thread int missing_weak_dyn_tls;

extern "C" int* missing_weak_dyn_tls_addr() {
  return &missing_weak_dyn_tls;
}
