/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
#if defined(WEBRTC_WIN)
// clang-format off
#include <windows.h>  // Must come first.
#include <mmsystem.h>
// clang-format on
#endif

#include <stdint.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "api/units/time_delta.h"
#include "rtc_base/event.h"
#include "rtc_base/task_queue_for_test.h"
#include "rtc_base/time_utils.h"
#include "test/gtest.h"

namespace webrtc {

namespace {

// Noop on all platforms except Windows, where it turns on high precision
// multimedia timers which increases the precision of TimeMillis() while in
// scope.
class EnableHighResTimers {
 public:
#if !defined(WEBRTC_WIN)
  EnableHighResTimers() {}
#else
  EnableHighResTimers() : enabled_(timeBeginPeriod(1) == TIMERR_NOERROR) {}
  ~EnableHighResTimers() {
    if (enabled_)
      timeEndPeriod(1);
  }

 private:
  const bool enabled_;
#endif
};

}  // namespace

// This task needs to be run manually due to the slowness of some of our bots.
// TODO(tommi): Can we run this on the perf bots?
TEST(TaskQueueTest, DISABLED_PostDelayedHighRes) {
  EnableHighResTimers high_res_scope;

  static const char kQueueName[] = "PostDelayedHighRes";
  rtc::Event event;
  TaskQueueForTest queue(kQueueName, TaskQueueFactory::Priority::HIGH);

  uint32_t start = rtc::TimeMillis();
  queue.PostDelayedTask(
      [&event, &queue] {
        EXPECT_TRUE(queue.IsCurrent());
        event.Set();
      },
      TimeDelta::Millis(3));
  EXPECT_TRUE(event.Wait(TimeDelta::Seconds(1)));
  uint32_t end = rtc::TimeMillis();
  // These tests are a little relaxed due to how "powerful" our test bots can
  // be.  Most recently we've seen windows bots fire the callback after 94-99ms,
  // which is why we have a little bit of leeway backwards as well.
  EXPECT_GE(end - start, 3u);
  EXPECT_NEAR(end - start, 3, 3u);
}

}  // namespace webrtc
