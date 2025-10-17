/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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
#include "rtc_base/synchronization/yield_policy.h"

#include <thread>  // Not allowed in production per Chromium style guide.

#include "rtc_base/event.h"
#include "test/gmock.h"
#include "test/gtest.h"

namespace rtc {
namespace {
class MockYieldHandler : public YieldInterface {
 public:
  MOCK_METHOD(void, YieldExecution, (), (override));
};
}  // namespace
TEST(YieldPolicyTest, HandlerReceivesYieldSignalWhenSet) {
  ::testing::StrictMock<MockYieldHandler> handler;
  {
    Event event;
    EXPECT_CALL(handler, YieldExecution()).Times(1);
    ScopedYieldPolicy policy(&handler);
    event.Set();
    event.Wait(Event::kForever);
  }
  {
    Event event;
    EXPECT_CALL(handler, YieldExecution()).Times(0);
    event.Set();
    event.Wait(Event::kForever);
  }
}

TEST(YieldPolicyTest, IsThreadLocal) {
  Event events[3];
  std::thread other_thread([&]() {
    ::testing::StrictMock<MockYieldHandler> local_handler;
    // The local handler is never called as we never Wait on this thread.
    EXPECT_CALL(local_handler, YieldExecution()).Times(0);
    ScopedYieldPolicy policy(&local_handler);
    events[0].Set();
    events[1].Set();
    events[2].Set();
  });

  // Waiting until the other thread has entered the scoped policy.
  events[0].Wait(Event::kForever);
  // Wait on this thread should not trigger the handler of that policy as it's
  // thread local.
  events[1].Wait(Event::kForever);

  // We can set a policy that's active on this thread independently.
  ::testing::StrictMock<MockYieldHandler> main_handler;
  EXPECT_CALL(main_handler, YieldExecution()).Times(1);
  ScopedYieldPolicy policy(&main_handler);
  events[2].Wait(Event::kForever);
  other_thread.join();
}
}  // namespace rtc
