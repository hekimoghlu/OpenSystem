/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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

extern "C" {
int __cxa_atexit(void (*func)(void*), void* arg, void* dso);

// TODO(b/175635923). __cxa_finalize's return type should actually be "void",
// but it is declared "int" here instead to be compatible with the declaration
// in an old version of cxxabi.h, which is included indirectly. The declarations
// of __cxa_atexit and __cxa_finalize are removed from newer versions of
// cxxabi.h, so once libc++ is updated, this return type should be changed to
// "void".
int __cxa_finalize(void* dso);
}

TEST(__cxa_atexit, simple) {
  int counter = 0;

  __cxa_atexit([](void* arg) { ++*static_cast<int*>(arg); }, &counter, &counter);

  __cxa_finalize(&counter);
  ASSERT_EQ(counter, 1);

  // The handler won't be called twice.
  __cxa_finalize(&counter);
  ASSERT_EQ(counter, 1);
}

TEST(__cxa_atexit, order) {
  static std::vector<int> actual;

  char handles[2];

  auto append_to_actual = [](void* arg) {
    int* idx = static_cast<int*>(arg);
    actual.push_back(*idx);
    delete idx;
  };

  for (int i = 0; i < 500; ++i) {
    __cxa_atexit(append_to_actual, new int{i}, &handles[i % 2]);
  }

  __cxa_finalize(&handles[0]);

  for (int i = 500; i < 750; ++i) {
    __cxa_atexit(append_to_actual, new int{i}, &handles[1]);
  }

  __cxa_finalize(&handles[1]);

  std::vector<int> expected;
  for (int i = 498; i >= 0; i -= 2) expected.push_back(i);
  for (int i = 749; i >= 500; --i) expected.push_back(i);
  for (int i = 499; i >= 1; i -= 2) expected.push_back(i);

  ASSERT_EQ(expected.size(), actual.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(expected[i], actual[i]) << "index=" << i;
  }
}
