/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#include <stddef.h>
#include <stdio.h>

#include <memory>
#include <random>

#include "rtc_base/checks.h"
#include "rtc_base/null_socket_server.h"
#include "rtc_base/thread.h"
#include "test/gtest.h"

namespace rtc {

namespace {

#if defined(MEMORY_SANITIZER)
void UseOfUninitializedValue() {
  int* buf = new int[2];
  std::random_device engine;
  if (buf[engine() % 2]) {  // Non-deterministic conditional.
    printf("Externally visible action.");
  }
  delete[] buf;
}

TEST(SanitizersDeathTest, MemorySanitizer) {
  EXPECT_DEATH(UseOfUninitializedValue(), "use-of-uninitialized-value");
}
#endif

#if defined(ADDRESS_SANITIZER)
void HeapUseAfterFree() {
  char* buf = new char[2];
  delete[] buf;
  buf[0] = buf[1];
}

TEST(SanitizersDeathTest, AddressSanitizer) {
  EXPECT_DEATH(HeapUseAfterFree(), "heap-use-after-free");
}
#endif

#if defined(UNDEFINED_SANITIZER)
// For ubsan:
void SignedIntegerOverflow() {
  int32_t x = 1234567890;
  x *= 2;
  (void)x;
}

// For ubsan_vptr:
struct Base {
  virtual void f() {}
  virtual ~Base() {}
};
struct Derived : public Base {};

void InvalidVptr() {
  Base b;
  auto* d = static_cast<Derived*>(&b);  // Bad downcast.
  d->f();  // Virtual function call with object of wrong dynamic type.
}

TEST(SanitizersDeathTest, UndefinedSanitizer) {
  EXPECT_DEATH(
      {
        SignedIntegerOverflow();
        InvalidVptr();
      },
      "runtime error");
}
#endif

#if defined(THREAD_SANITIZER)
class IncrementThread : public Thread {
 public:
  explicit IncrementThread(int* value)
      : Thread(std::make_unique<NullSocketServer>()), value_(value) {}

  IncrementThread(const IncrementThread&) = delete;
  IncrementThread& operator=(const IncrementThread&) = delete;

  void Run() override {
    ++*value_;
    Thread::Current()->SleepMs(100);
  }

  // Un-protect Thread::Join for the test.
  void Join() { Thread::Join(); }

 private:
  int* value_;
};

void DataRace() {
  int value = 0;
  IncrementThread thread1(&value);
  IncrementThread thread2(&value);
  thread1.Start();
  thread2.Start();
  thread1.Join();
  thread2.Join();
  // TSan seems to mess with gtest's death detection.
  // Fail intentionally, and rely on detecting the error message.
  RTC_CHECK_NOTREACHED();
}

TEST(SanitizersDeathTest, ThreadSanitizer) {
  EXPECT_DEATH(DataRace(), "data race");
}
#endif

}  // namespace

}  // namespace rtc
