/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
#ifndef TEST_TIME_CONTROLLER_SIMULATED_THREAD_H_
#define TEST_TIME_CONTROLLER_SIMULATED_THREAD_H_

#include <memory>

#include "rtc_base/synchronization/mutex.h"
#include "test/time_controller/simulated_time_controller.h"

namespace webrtc {

class SimulatedThread : public rtc::Thread,
                        public sim_time_impl::SimulatedSequenceRunner {
 public:
  using CurrentThreadSetter = CurrentThreadSetter;
  SimulatedThread(sim_time_impl::SimulatedTimeControllerImpl* handler,
                  absl::string_view name,
                  std::unique_ptr<rtc::SocketServer> socket_server);
  ~SimulatedThread() override;

  void RunReady(Timestamp at_time) override;

  Timestamp GetNextRunTime() const override {
    MutexLock lock(&lock_);
    return next_run_time_;
  }

  TaskQueueBase* GetAsTaskQueue() override { return this; }

  // Thread interface
  void BlockingCallImpl(rtc::FunctionView<void()> functor,
                        const Location& location) override;
  void PostTaskImpl(absl::AnyInvocable<void() &&> task,
                    const PostTaskTraits& traits,
                    const Location& location) override;
  void PostDelayedTaskImpl(absl::AnyInvocable<void() &&> task,
                           TimeDelta delay,
                           const PostDelayedTaskTraits& traits,
                           const Location& location) override;

  void Stop() override;

 private:
  sim_time_impl::SimulatedTimeControllerImpl* const handler_;
  // Using char* to be debugger friendly.
  char* name_;
  mutable Mutex lock_;
  Timestamp next_run_time_ RTC_GUARDED_BY(lock_) = Timestamp::PlusInfinity();
};

class SimulatedMainThread : public SimulatedThread {
 public:
  explicit SimulatedMainThread(
      sim_time_impl::SimulatedTimeControllerImpl* handler);
  ~SimulatedMainThread();

 private:
  CurrentThreadSetter current_setter_;
};
}  // namespace webrtc
#endif  // TEST_TIME_CONTROLLER_SIMULATED_THREAD_H_
