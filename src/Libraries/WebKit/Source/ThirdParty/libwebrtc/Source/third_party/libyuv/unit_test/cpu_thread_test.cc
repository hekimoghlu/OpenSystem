/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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

#include "libyuv/cpu_id.h"

#if defined(__clang__) && !defined(__wasm__)
#if __has_include(<pthread.h>)
#define LIBYUV_HAVE_PTHREAD 1
#endif
#elif defined(__linux__)
#define LIBYUV_HAVE_PTHREAD 1
#endif

#ifdef LIBYUV_HAVE_PTHREAD
#include <pthread.h>
#endif

namespace libyuv {

#ifdef LIBYUV_HAVE_PTHREAD
static void* ThreadMain(void* arg) {
  int* flags = static_cast<int*>(arg);

  *flags = TestCpuFlag(kCpuInitialized);
  return nullptr;
}
#endif  // LIBYUV_HAVE_PTHREAD

// Call TestCpuFlag() from two threads. ThreadSanitizer should not report any
// data race.
TEST(LibYUVCpuThreadTest, TestCpuFlagMultipleThreads) {
#ifdef LIBYUV_HAVE_PTHREAD
  int cpu_flags1;
  int cpu_flags2;
  int ret;
  pthread_t thread1;
  pthread_t thread2;

  MaskCpuFlags(0);  // Reset to 0 to allow auto detect.
  ret = pthread_create(&thread1, nullptr, ThreadMain, &cpu_flags1);
  ASSERT_EQ(ret, 0);
  ret = pthread_create(&thread2, nullptr, ThreadMain, &cpu_flags2);
  ASSERT_EQ(ret, 0);
  ret = pthread_join(thread1, nullptr);
  EXPECT_EQ(ret, 0);
  ret = pthread_join(thread2, nullptr);
  EXPECT_EQ(ret, 0);
  EXPECT_EQ(cpu_flags1, cpu_flags2);
#else
  printf("pthread unavailable; Test skipped.");
#endif  // LIBYUV_HAVE_PTHREAD
}

}  // namespace libyuv
