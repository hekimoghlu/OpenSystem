/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
#include <cxxabi.h>
#include <gtest/gtest.h>
#include <string.h>

TEST(__cxa_demangle, cxa_demangle_fuzz_152588929) {
#if defined(__aarch64__)
  // Test the C++ demangler on an invalid mangled string. libc++abi currently
  // parses it like so:
  //    (1 "\006") (I (L e "eeEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE" E) E)
  // There are a few interesting things about this mangled input:
  //  - The IA64 C++ ABI specifies that an FP literal's hex chars are lowercase.
  //    The libc++abi demangler currently accepts uppercase A-F digits, which is
  //    confusing because 'E' is supposed to mark the end of the <expr-primary>.
  //  - libc++abi uses snprintf("%a") which puts an unspecified number of bits
  //    in the digit before the decimal point.
  //  - The identifier name is "\006", and the IA64 C++ ABI spec is explicit
  //    about not specifying the encoding for characters outside of
  //    [_A-Za-z0-9].
  //  - The 'e' type is documented as "long double, __float80", and in practice
  //    the length of the literal depends on the arch. For arm64, it is a
  //    128-bit FP type encoded using 32 hex chars. The situation with x86-64
  //    Android OTOH is messy because Clang uses 'g' for its 128-bit
  //    long double.
  char* p = abi::__cxa_demangle("1\006ILeeeEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE", 0, 0, 0);
  if (p && !strcmp(p, "\x6<-0x1.cecececececececececececececep+11983")) {
    // Prior to toolchain.org/D77924, libc++abi left off the "L>" suffix.
  } else if (p && !strcmp(p, "\x6<-0x1.cecececececececececececececep+11983L>")) {
    // After toolchain.org/D77924, the "L>" suffix is present. libc++abi
    // accepts A-F digits but decodes each using (digit - 'a' + 10), turning 'E'
    // into -18.
  } else {
    // TODO: Remove the other accepted outputs, because libc++abi probably
    // should reject this input.
    ASSERT_EQ(nullptr, p) << p;
  }
  free(p);
#endif
}

TEST(__cxa_demangle, DISABLED_cxa_demangle_fuzz_167977068) {
#if defined(__aarch64__)
  char* p = abi::__cxa_demangle("DTLeeeeeeeeeeeeeeeeeeeeeeeeeEEEEeeEEEE", 0, 0, 0);
  ASSERT_EQ(nullptr, p) << p;
  free(p);
#endif
}
