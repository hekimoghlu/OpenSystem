/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
#include <dlfcn.h>
#include <stdlib.h>

#include <gtest/gtest.h>

#include <android-base/silent_death_test.h>
#include <android-base/test_utils.h>

using HwasanDeathTest = SilentDeathTest;


#ifdef HWASAN_TEST_STATIC
#define MAYBE_DlopenAbsolutePath DISABLED_DlopenAbsolutePath
// TODO(fmayer): figure out why uaf is misclassified as out of bounds for
// static executables.
#define MAYBE_UseAfterFree DISABLED_UseAfterFree
#else
#define MAYBE_DlopenAbsolutePath DlopenAbsolutePath
#define MAYBE_UseAfterFree UseAfterFree
#endif

TEST_F(HwasanDeathTest, MAYBE_UseAfterFree) {
  EXPECT_DEATH(
      {
        void* m = malloc(1);
        volatile char* x = const_cast<volatile char*>(reinterpret_cast<char*>(m));
        *x = 1;
        free(m);
        *x = 2;
      },
      "use-after-free");
}

TEST_F(HwasanDeathTest, OutOfBounds) {
  EXPECT_DEATH(
      {
        void* m = malloc(1);
        volatile char* x = const_cast<volatile char*>(reinterpret_cast<char*>(m));
        x[1] = 1;
      },
      "buffer-overflow");
}

// Check whether dlopen of /foo/bar.so checks /foo/hwasan/bar.so first.
TEST(HwasanTest, MAYBE_DlopenAbsolutePath) {
  std::string path = android::base::GetExecutableDirectory() + "/libtest_simple_hwasan.so";
  ASSERT_EQ(0, access(path.c_str(), F_OK));  // Verify test setup.
  std::string hwasan_path =
      android::base::GetExecutableDirectory() + "/hwasan/libtest_simple_hwasan.so";
  ASSERT_EQ(0, access(hwasan_path.c_str(), F_OK));  // Verify test setup.

  void* handle = dlopen(path.c_str(), RTLD_NOW);
  ASSERT_TRUE(handle != nullptr);
  uint32_t* compiled_with_hwasan =
      reinterpret_cast<uint32_t*>(dlsym(handle, "dlopen_testlib_compiled_with_hwasan"));
  EXPECT_TRUE(*compiled_with_hwasan);
  dlclose(handle);
}

TEST(HwasanTest, IsRunningWithHWasan) {
  EXPECT_TRUE(running_with_hwasan());
}
