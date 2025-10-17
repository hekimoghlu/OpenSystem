/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 4, 2024.
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
#include <err.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <map>
#include <vector>

#include <gtest/gtest.h>

struct FdtrackFrame {
  const char* function_name;
  uint64_t function_offset;
};

std::map<int, std::vector<FdtrackFrame>> RunFdtrack(std::function<void()> f) {
  void* libfdtrack = dlopen("libfdtrack.so", RTLD_NOW);
  if (!libfdtrack) {
    errx(1, "failed to dlopen libfdtrack.so: %s", dlerror());
  }

  using fdtrack_callback_t = bool (*)(int fd, const char* const* function_names,
                                      const uint64_t* function_offsets, size_t count, void* arg);
  auto fdtrack_iterate = reinterpret_cast<void (*)(fdtrack_callback_t, void* arg)>(
      dlsym(libfdtrack, "fdtrack_iterate"));
  if (!fdtrack_iterate) {
    errx(1, "failed to dlsym fdtrack_iterate");
  }

  f();

  std::map<int, std::vector<FdtrackFrame>> result;
  fdtrack_iterate(
      [](int fd, const char* const* function_names, const uint64_t* function_offsets, size_t count,
         void* arg) {
        auto& map = *static_cast<decltype(result)*>(arg);
        for (size_t i = 0; i < count; ++i) {
          map[fd].push_back(FdtrackFrame{
              .function_name = function_names[i],
              .function_offset = function_offsets[i],
          });
        }

        return true;
      },
      &result);

  return result;
}

TEST(fdtrack, open) {
  static int fd = -1;
  auto result = RunFdtrack([]() { fd = open("/dev/null", O_RDONLY | O_CLOEXEC); });

  ASSERT_NE(-1, fd);
  ASSERT_EQ(1, result.size());
  ASSERT_EQ(fd, result.begin()->first);
  ASSERT_NE(nullptr, strstr(result.begin()->second.at(0).function_name, "open"));
}

TEST(fdtrack, close) {
  static int fd1 = -1;
  static int fd2 = -1;
  static int fd3 = -1;
  auto result = RunFdtrack([]() {
    fd1 = open("/dev/null", O_RDONLY | O_CLOEXEC);
    fd2 = open("/dev/null", O_RDONLY | O_CLOEXEC);
    fd3 = open("/dev/null", O_RDONLY | O_CLOEXEC);
    close(fd2);
  });

  ASSERT_NE(-1, fd1);
  ASSERT_NE(-1, fd2);
  ASSERT_NE(-1, fd3);

  ASSERT_EQ(2, result.size());
  ASSERT_EQ(1, result.count(fd1));
  ASSERT_EQ(1, result.count(fd3));

  ASSERT_NE(nullptr, strstr(result[fd1].at(0).function_name, "open"));
  ASSERT_NE(nullptr, strstr(result[fd3].at(0).function_name, "open"));
}
