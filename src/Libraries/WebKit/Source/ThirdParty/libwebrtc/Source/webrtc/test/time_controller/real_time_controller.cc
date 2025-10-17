/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#include "test/time_controller/real_time_controller.h"

#include "api/field_trials_view.h"
#include "api/task_queue/default_task_queue_factory.h"
#include "rtc_base/null_socket_server.h"

namespace webrtc {
namespace {
class MainThread : public rtc::Thread {
 public:
  MainThread()
      : Thread(std::make_unique<rtc::NullSocketServer>(), false),
        current_setter_(this) {
    DoInit();
  }
  ~MainThread() {
    Stop();
    DoDestroy();
  }

 private:
  CurrentThreadSetter current_setter_;
};
}  // namespace
RealTimeController::RealTimeController(const FieldTrialsView* field_trials)
    : task_queue_factory_(CreateDefaultTaskQueueFactory(field_trials)),
      main_thread_(std::make_unique<MainThread>()) {
  main_thread_->SetName("Main", this);
}

Clock* RealTimeController::GetClock() {
  return Clock::GetRealTimeClock();
}

TaskQueueFactory* RealTimeController::GetTaskQueueFactory() {
  return task_queue_factory_.get();
}

std::unique_ptr<rtc::Thread> RealTimeController::CreateThread(
    const std::string& name,
    std::unique_ptr<rtc::SocketServer> socket_server) {
  if (!socket_server)
    socket_server = std::make_unique<rtc::NullSocketServer>();
  auto res = std::make_unique<rtc::Thread>(std::move(socket_server));
  res->SetName(name, nullptr);
  res->Start();
  return res;
}

rtc::Thread* RealTimeController::GetMainThread() {
  return main_thread_.get();
}

void RealTimeController::AdvanceTime(TimeDelta duration) {
  main_thread_->ProcessMessages(duration.ms());
}

}  // namespace webrtc
