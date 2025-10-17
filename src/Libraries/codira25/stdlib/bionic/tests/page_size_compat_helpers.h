/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 2, 2024.
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

#include "elf_max_page_size.h"
#include "gtest_globals.h"

#include <android-base/stringprintf.h>

#include <string>

#include <dlfcn.h>
#include <gtest/gtest.h>
#include <unistd.h>

static inline void OpenTestLibrary(std::string lib, bool expect_fail, void** handle) {
  void* _handle = dlopen(lib.c_str(), RTLD_NODELETE);
  const char* dlopen_error = dlerror();

  if (expect_fail) {
    ASSERT_EQ(_handle, nullptr);

    const std::string expected_error = android::base::StringPrintf(
        "dlopen failed: \"%s\" program alignment (%d) cannot be smaller than system page size (%d)",
        lib.c_str(), 4096, getpagesize());

    ASSERT_EQ(expected_error, dlopen_error);
  } else {
    ASSERT_NE(_handle, nullptr) << "Failed to dlopen shared library \"" << lib
                                << "\": " << dlopen_error;
  }

  *handle = _handle;
}

static inline void CallTestFunction(void* handle) {
  loader_test_func_t loader_test_func = (loader_test_func_t)dlsym(handle, "loader_test_func");
  const char* dlsym_error = dlerror();

  ASSERT_EQ(dlsym_error, nullptr) << "Failed to locate symbol \"loader_test_func\": "
                                  << dlsym_error;

  int res = loader_test_func();
  ASSERT_EQ(res, TEST_RESULT_BASE + TEST_RESULT_INCREMENT);

  // Call loader_test_func() twice to ensure we can modify writeable data and bss data
  res = loader_test_func();
  ASSERT_EQ(res, TEST_RESULT_BASE + (2 * TEST_RESULT_INCREMENT));
}
