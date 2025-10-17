/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#include <dlfcn.h>
#include <stdlib.h>

#include <android-base/logging.h>
#include <gtest/gtest.h>

static size_t NumberBuffers() {
  size_t bufs = 0;
  std::ifstream file("/proc/self/maps");
  CHECK(file.is_open());
  std::string line;
  while (std::getline(file, line)) {
    if (line.find("stack_mte_ring") != std::string::npos) {
      ++bufs;
    }
  }
  return bufs;
}

static size_t NumberThreads() {
  std::filesystem::directory_iterator di("/proc/self/task");
  return std::distance(begin(di), end(di));
}

TEST(MemtagStackAbiTest, MainThread) {
#if defined(__BIONIC__) && defined(__aarch64__)
  ASSERT_EQ(NumberBuffers(), 1U);
  ASSERT_EQ(NumberBuffers(), NumberThreads());
#else
  GTEST_SKIP() << "requires bionic arm64";
#endif
}

TEST(MemtagStackAbiTest, JoinableThread) {
#if defined(__BIONIC__) && defined(__aarch64__)
  ASSERT_EQ(NumberBuffers(), 1U);
  ASSERT_EQ(NumberBuffers(), NumberThreads());
  std::thread th([] {
    ASSERT_EQ(NumberBuffers(), 2U);
    ASSERT_EQ(NumberBuffers(), NumberThreads());
  });
  th.join();
  ASSERT_EQ(NumberBuffers(), 1U);
  ASSERT_EQ(NumberBuffers(), NumberThreads());
#else
  GTEST_SKIP() << "requires bionic arm64";
#endif
}

TEST(MemtagStackAbiTest, DetachedThread) {
#if defined(__BIONIC__) && defined(__aarch64__)
  ASSERT_EQ(NumberBuffers(), 1U);
  ASSERT_EQ(NumberBuffers(), NumberThreads());
  std::thread th([] {
    ASSERT_EQ(NumberBuffers(), 2U);
    ASSERT_EQ(NumberBuffers(), NumberThreads());
  });
  th.detach();
  // Leave the thread some time to exit.
  for (int i = 0; NumberBuffers() != 1 && i < 3; ++i) {
    sleep(1);
  }
  ASSERT_EQ(NumberBuffers(), 1U);
  ASSERT_EQ(NumberBuffers(), NumberThreads());
#else
  GTEST_SKIP() << "requires bionic arm64";
#endif
}
