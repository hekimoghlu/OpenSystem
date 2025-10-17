/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 5, 2024.
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
#include "rtc_base/task_queue_stdlib.h"

#include "api/task_queue/task_queue_factory.h"
#include "api/task_queue/task_queue_test.h"
#include "api/units/time_delta.h"
#include "rtc_base/event.h"
#include "rtc_base/logging.h"
#include "system_wrappers/include/sleep.h"
#include "test/gtest.h"

namespace webrtc {
namespace {

std::unique_ptr<TaskQueueFactory> CreateTaskQueueFactory(
    const webrtc::FieldTrialsView*) {
  return CreateTaskQueueStdlibFactory();
}

INSTANTIATE_TEST_SUITE_P(TaskQueueStdlib,
                         TaskQueueTest,
                         ::testing::Values(CreateTaskQueueFactory));

class StringPtrLogSink : public rtc::LogSink {
 public:
  explicit StringPtrLogSink(std::string* log_data) : log_data_(log_data) {}

 private:
  void OnLogMessage(const std::string& message) override {
    OnLogMessage(absl::string_view(message));
  }
  void OnLogMessage(absl::string_view message) override {
    log_data_->append(message.begin(), message.end());
  }
  std::string* const log_data_;
};

TEST(TaskQueueStdlib, AvoidsSpammingLogOnInactivity) {
  std::string log_output;
  StringPtrLogSink stream(&log_output);
  rtc::LogMessage::AddLogToStream(&stream, rtc::LS_VERBOSE);
  auto task_queue = CreateTaskQueueStdlibFactory()->CreateTaskQueue(
      "test", TaskQueueFactory::Priority::NORMAL);
  auto wait_duration = rtc::Event::kDefaultWarnDuration + TimeDelta::Seconds(1);
  SleepMs(wait_duration.ms());
  EXPECT_EQ(log_output.length(), 0u);
  task_queue = nullptr;
  rtc::LogMessage::RemoveLogToStream(&stream);
}

}  // namespace
}  // namespace webrtc
