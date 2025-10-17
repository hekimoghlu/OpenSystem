/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#include <gtest/gtest.h>

#include <thread>

#include "gtest_globals.h"
#include "utils.h"

// Specify the LE access model explicitly. This file is compiled into the
// bionic-unit-tests executable, but the compiler sees an -fpic object file
// output into a static library, so it defaults to dynamic TLS accesses.

// This variable will be zero-initialized (.tbss)
__attribute__((tls_model("local-exec"))) static __thread int tlsvar_le_zero;

// This variable will have an initializer (.tdata)
__attribute__((tls_model("local-exec"))) static __thread int tlsvar_le_init = 10;

// Access libtest_elftls_shared_var's TLS variable using an IE access.
__attribute__((tls_model("initial-exec"))) extern "C" __thread int elftls_shared_var;

TEST(elftls, basic_le) {
  // Check the variables on the main thread.
  ASSERT_EQ(11, ++tlsvar_le_init);
  ASSERT_EQ(1, ++tlsvar_le_zero);

  // Check variables on a new thread.
  std::thread([] {
    ASSERT_EQ(11, ++tlsvar_le_init);
    ASSERT_EQ(1, ++tlsvar_le_zero);
  }).join();
}

TEST(elftls, shared_ie) {
  ASSERT_EQ(21, ++elftls_shared_var);
  std::thread([] {
    ASSERT_EQ(21, ++elftls_shared_var);
  }).join();
}

extern "C" int bump_static_tls_var_1();
extern "C" int bump_static_tls_var_2();

TEST(elftls, tprel_addend) {
  ASSERT_EQ(4, bump_static_tls_var_1());
  ASSERT_EQ(8, bump_static_tls_var_2());
  std::thread([] {
    ASSERT_EQ(4, bump_static_tls_var_1());
    ASSERT_EQ(8, bump_static_tls_var_2());
  }).join();
}

// Because this C++ source file is built with -fpic, the compiler will access
// this variable using a GD model. Typically, the static linker will relax the
// GD to LE, but the arm32 linker doesn't do TLS relaxations, so we can test
// calling __tls_get_addr in a static executable. The static linker knows that
// the main executable's TlsIndex::module_id is 1 and writes that into the GOT.
__thread int tlsvar_general = 30;

TEST(elftls, general) {
  ASSERT_EQ(31, ++tlsvar_general);
  std::thread([] {
    ASSERT_EQ(31, ++tlsvar_general);
  }).join();
}

TEST(elftls, align_test) {
  std::string helper = GetTestLibRoot() + "/elftls_align_test_helper";
  ExecTestHelper eth;
  eth.SetArgs({helper.c_str(), nullptr});
  eth.Run([&]() { execve(helper.c_str(), eth.GetArgs(), eth.GetEnv()); }, 0, nullptr);
}

TEST(elftls, skew_align_test) {
  std::string helper = GetTestLibRoot() + "/elftls_skew_align_test_helper";
  ExecTestHelper eth;
  eth.SetArgs({helper.c_str(), nullptr});
  eth.Run([&]() { execve(helper.c_str(), eth.GetArgs(), eth.GetEnv()); }, 0, nullptr);
}
