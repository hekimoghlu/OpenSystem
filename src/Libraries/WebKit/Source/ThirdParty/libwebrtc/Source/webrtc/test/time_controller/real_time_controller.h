/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#ifndef TEST_TIME_CONTROLLER_REAL_TIME_CONTROLLER_H_
#define TEST_TIME_CONTROLLER_REAL_TIME_CONTROLLER_H_

#include <functional>
#include <memory>

#include "api/field_trials_view.h"
#include "api/task_queue/task_queue_factory.h"
#include "api/test/time_controller.h"
#include "api/units/time_delta.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {
class RealTimeController : public TimeController {
 public:
  RealTimeController(const FieldTrialsView* field_trials = nullptr);

  Clock* GetClock() override;
  TaskQueueFactory* GetTaskQueueFactory() override;
  std::unique_ptr<rtc::Thread> CreateThread(
      const std::string& name,
      std::unique_ptr<rtc::SocketServer> socket_server) override;
  rtc::Thread* GetMainThread() override;
  void AdvanceTime(TimeDelta duration) override;

 private:
  const std::unique_ptr<TaskQueueFactory> task_queue_factory_;
  const std::unique_ptr<rtc::Thread> main_thread_;
};

}  // namespace webrtc

#endif  // TEST_TIME_CONTROLLER_REAL_TIME_CONTROLLER_H_
