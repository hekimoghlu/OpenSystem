/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#include "rtc_base/platform_thread.h"

#include <optional>

#include "rtc_base/event.h"
#include "system_wrappers/include/sleep.h"
#include "test/gmock.h"

namespace rtc {

TEST(PlatformThreadTest, DefaultConstructedIsEmpty) {
  PlatformThread thread;
  EXPECT_EQ(thread.GetHandle(), std::nullopt);
  EXPECT_TRUE(thread.empty());
}

TEST(PlatformThreadTest, StartFinalize) {
  PlatformThread thread = PlatformThread::SpawnJoinable([] {}, "1");
  EXPECT_NE(thread.GetHandle(), std::nullopt);
  EXPECT_FALSE(thread.empty());
  thread.Finalize();
  EXPECT_TRUE(thread.empty());
  rtc::Event done;
  thread = PlatformThread::SpawnDetached([&] { done.Set(); }, "2");
  EXPECT_FALSE(thread.empty());
  thread.Finalize();
  EXPECT_TRUE(thread.empty());
  done.Wait(webrtc::TimeDelta::Seconds(30));
}

TEST(PlatformThreadTest, MovesEmpty) {
  PlatformThread thread1;
  PlatformThread thread2 = std::move(thread1);
  EXPECT_TRUE(thread1.empty());
  EXPECT_TRUE(thread2.empty());
}

TEST(PlatformThreadTest, MovesHandles) {
  PlatformThread thread1 = PlatformThread::SpawnJoinable([] {}, "1");
  PlatformThread thread2 = std::move(thread1);
  EXPECT_TRUE(thread1.empty());
  EXPECT_FALSE(thread2.empty());
  rtc::Event done;
  thread1 = PlatformThread::SpawnDetached([&] { done.Set(); }, "2");
  thread2 = std::move(thread1);
  EXPECT_TRUE(thread1.empty());
  EXPECT_FALSE(thread2.empty());
  done.Wait(webrtc::TimeDelta::Seconds(30));
}

TEST(PlatformThreadTest,
     TwoThreadHandlesAreDifferentWhenStartedAndEqualWhenJoined) {
  PlatformThread thread1 = PlatformThread();
  PlatformThread thread2 = PlatformThread();
  EXPECT_EQ(thread1.GetHandle(), thread2.GetHandle());
  thread1 = PlatformThread::SpawnJoinable([] {}, "1");
  thread2 = PlatformThread::SpawnJoinable([] {}, "2");
  EXPECT_NE(thread1.GetHandle(), thread2.GetHandle());
  thread1.Finalize();
  EXPECT_NE(thread1.GetHandle(), thread2.GetHandle());
  thread2.Finalize();
  EXPECT_EQ(thread1.GetHandle(), thread2.GetHandle());
}

TEST(PlatformThreadTest, RunFunctionIsCalled) {
  bool flag = false;
  PlatformThread::SpawnJoinable([&] { flag = true; }, "T");
  EXPECT_TRUE(flag);
}

TEST(PlatformThreadTest, JoinsThread) {
  // This test flakes if there are problems with the join implementation.
  rtc::Event event;
  PlatformThread::SpawnJoinable([&] { event.Set(); }, "T");
  EXPECT_TRUE(event.Wait(/*give_up_after=*/webrtc::TimeDelta::Zero()));
}

TEST(PlatformThreadTest, StopsBeforeDetachedThreadExits) {
  // This test flakes if there are problems with the detached thread
  // implementation.
  bool flag = false;
  rtc::Event thread_started;
  rtc::Event thread_continue;
  rtc::Event thread_exiting;
  PlatformThread::SpawnDetached(
      [&] {
        thread_started.Set();
        thread_continue.Wait(Event::kForever);
        flag = true;
        thread_exiting.Set();
      },
      "T");
  thread_started.Wait(Event::kForever);
  EXPECT_FALSE(flag);
  thread_continue.Set();
  thread_exiting.Wait(Event::kForever);
  EXPECT_TRUE(flag);
}

}  // namespace rtc
