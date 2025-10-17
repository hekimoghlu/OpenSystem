/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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
#include "test/run_loop.h"

#include "api/task_queue/task_queue_base.h"
#include "api/units/time_delta.h"
#include "test/gtest.h"

namespace webrtc {

TEST(RunLoopTest, TaskQueueOnThread) {
  test::RunLoop loop;
  EXPECT_EQ(TaskQueueBase::Current(), loop.task_queue());
  EXPECT_TRUE(loop.task_queue()->IsCurrent());
}

TEST(RunLoopTest, Flush) {
  test::RunLoop loop;
  int counter = 0;
  loop.PostTask([&counter]() { ++counter; });
  EXPECT_EQ(counter, 0);
  loop.Flush();
  EXPECT_EQ(counter, 1);
}

TEST(RunLoopTest, Delayed) {
  test::RunLoop loop;
  bool ran = false;
  loop.task_queue()->PostDelayedTask(
      [&ran, &loop]() {
        ran = true;
        loop.Quit();
      },
      TimeDelta::Millis(100));
  loop.Flush();
  EXPECT_FALSE(ran);
  loop.Run();
  EXPECT_TRUE(ran);
}

TEST(RunLoopTest, PostAndQuit) {
  test::RunLoop loop;
  bool ran = false;
  loop.PostTask([&ran, &loop]() {
    ran = true;
    loop.Quit();
  });
  loop.Run();
  EXPECT_TRUE(ran);
}

}  // namespace webrtc
