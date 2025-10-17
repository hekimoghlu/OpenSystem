/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
#include "rtc_base/synchronization/mutex.h"

#include <stddef.h>
#include <stdint.h>

#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "rtc_base/checks.h"
#include "rtc_base/event.h"
#include "rtc_base/platform_thread.h"
#include "rtc_base/synchronization/yield.h"
#include "rtc_base/thread.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

using ::rtc::Event;
using ::rtc::Thread;

constexpr int kNumThreads = 16;

template <class MutexType>
class RTC_LOCKABLE RawMutexLocker {
 public:
  explicit RawMutexLocker(MutexType& mutex) : mutex_(mutex) {}
  void Lock() RTC_EXCLUSIVE_LOCK_FUNCTION() { mutex_.Lock(); }
  void Unlock() RTC_UNLOCK_FUNCTION() { mutex_.Unlock(); }

 private:
  MutexType& mutex_;
};

class RTC_LOCKABLE RawMutexTryLocker {
 public:
  explicit RawMutexTryLocker(Mutex& mutex) : mutex_(mutex) {}
  void Lock() RTC_EXCLUSIVE_LOCK_FUNCTION() {
    while (!mutex_.TryLock()) {
      YieldCurrentThread();
    }
  }
  void Unlock() RTC_UNLOCK_FUNCTION() { mutex_.Unlock(); }

 private:
  Mutex& mutex_;
};

template <class MutexType, class MutexLockType>
class MutexLockLocker {
 public:
  explicit MutexLockLocker(MutexType& mutex) : mutex_(mutex) {}
  void Lock() { lock_ = std::make_unique<MutexLockType>(&mutex_); }
  void Unlock() { lock_ = nullptr; }

 private:
  MutexType& mutex_;
  std::unique_ptr<MutexLockType> lock_;
};

template <class MutexType, class MutexLocker>
class LockRunner {
 public:
  template <typename... Args>
  explicit LockRunner(Args... args)
      : threads_active_(0),
        start_event_(true, false),
        done_event_(true, false),
        shared_value_(0),
        mutex_(args...),
        locker_(mutex_) {}

  bool Run() {
    // Signal all threads to start.
    start_event_.Set();

    // Wait for all threads to finish.
    return done_event_.Wait(kLongTime);
  }

  void SetExpectedThreadCount(int count) { threads_active_ = count; }

  int shared_value() {
    int shared_value;
    locker_.Lock();
    shared_value = shared_value_;
    locker_.Unlock();
    return shared_value;
  }

  void Loop() {
    ASSERT_TRUE(start_event_.Wait(kLongTime));
    locker_.Lock();

    EXPECT_EQ(0, shared_value_);
    int old = shared_value_;

    // Use a loop to increase the chance of race. If the `locker_`
    // implementation is faulty, it would be improbable that the error slips
    // through.
    for (int i = 0; i < kOperationsToRun; ++i) {
      benchmark::DoNotOptimize(++shared_value_);
    }
    EXPECT_EQ(old + kOperationsToRun, shared_value_);
    shared_value_ = 0;

    locker_.Unlock();
    if (threads_active_.fetch_sub(1) == 1) {
      done_event_.Set();
    }
  }

 private:
  static constexpr TimeDelta kLongTime = TimeDelta::Seconds(10);
  static constexpr int kOperationsToRun = 1000;

  std::atomic<int> threads_active_;
  Event start_event_;
  Event done_event_;
  int shared_value_;
  MutexType mutex_;
  MutexLocker locker_;
};

template <typename Runner>
void StartThreads(std::vector<std::unique_ptr<Thread>>& threads,
                  Runner* handler) {
  for (int i = 0; i < kNumThreads; ++i) {
    std::unique_ptr<Thread> thread(Thread::Create());
    thread->Start();
    thread->PostTask([handler] { handler->Loop(); });
    threads.push_back(std::move(thread));
  }
}

TEST(MutexTest, ProtectsSharedResourceWithMutexAndRawMutexLocker) {
  std::vector<std::unique_ptr<Thread>> threads;
  LockRunner<Mutex, RawMutexLocker<Mutex>> runner;
  StartThreads(threads, &runner);
  runner.SetExpectedThreadCount(kNumThreads);
  EXPECT_TRUE(runner.Run());
  EXPECT_EQ(0, runner.shared_value());
}

TEST(MutexTest, ProtectsSharedResourceWithMutexAndRawMutexTryLocker) {
  std::vector<std::unique_ptr<Thread>> threads;
  LockRunner<Mutex, RawMutexTryLocker> runner;
  StartThreads(threads, &runner);
  runner.SetExpectedThreadCount(kNumThreads);
  EXPECT_TRUE(runner.Run());
  EXPECT_EQ(0, runner.shared_value());
}

TEST(MutexTest, ProtectsSharedResourceWithMutexAndMutexLocker) {
  std::vector<std::unique_ptr<Thread>> threads;
  LockRunner<Mutex, MutexLockLocker<Mutex, MutexLock>> runner;
  StartThreads(threads, &runner);
  runner.SetExpectedThreadCount(kNumThreads);
  EXPECT_TRUE(runner.Run());
  EXPECT_EQ(0, runner.shared_value());
}

}  // namespace
}  // namespace webrtc
