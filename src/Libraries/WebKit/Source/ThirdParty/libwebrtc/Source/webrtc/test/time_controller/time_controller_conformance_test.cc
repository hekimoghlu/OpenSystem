/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 22, 2023.
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
#include <memory>
#include <vector>

#include "api/test/time_controller.h"
#include "api/units/time_delta.h"
#include "rtc_base/event.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/thread.h"
#include "rtc_base/thread_annotations.h"
#include "test/gmock.h"
#include "test/gtest.h"
#include "test/time_controller/real_time_controller.h"
#include "test/time_controller/simulated_time_controller.h"

namespace webrtc {
namespace {

using ::testing::ElementsAreArray;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

enum class TimeMode { kRealTime, kSimulated };

std::unique_ptr<TimeController> CreateTimeController(TimeMode mode) {
  switch (mode) {
    case TimeMode::kRealTime:
      return std::make_unique<RealTimeController>();
    case TimeMode::kSimulated:
      // Using an offset of 100000 to get nice fixed width and readable
      // timestamps in typical test scenarios.
      constexpr Timestamp kSimulatedStartTime = Timestamp::Seconds(100000);
      return std::make_unique<GlobalSimulatedTimeController>(
          kSimulatedStartTime);
  }
}

std::string ParamsToString(const TestParamInfo<webrtc::TimeMode>& param) {
  switch (param.param) {
    case webrtc::TimeMode::kRealTime:
      return "RealTime";
    case webrtc::TimeMode::kSimulated:
      return "SimulatedTime";
    default:
      RTC_DCHECK_NOTREACHED() << "Time mode not supported";
  }
}

// Keeps order of executions. May be called from different threads.
class ExecutionOrderKeeper {
 public:
  void Executed(int execution_id) {
    MutexLock lock(&mutex_);
    order_.push_back(execution_id);
  }

  std::vector<int> order() const {
    MutexLock lock(&mutex_);
    return order_;
  }

 private:
  mutable Mutex mutex_;
  std::vector<int> order_ RTC_GUARDED_BY(mutex_);
};

// Tests conformance between real time and simulated time time controller.
class SimulatedRealTimeControllerConformanceTest
    : public TestWithParam<webrtc::TimeMode> {};

TEST_P(SimulatedRealTimeControllerConformanceTest, ThreadPostOrderTest) {
  std::unique_ptr<TimeController> time_controller =
      CreateTimeController(GetParam());
  std::unique_ptr<rtc::Thread> thread = time_controller->CreateThread("thread");

  // Tasks on thread have to be executed in order in which they were
  // posted.
  ExecutionOrderKeeper execution_order;
  thread->PostTask([&]() { execution_order.Executed(1); });
  thread->PostTask([&]() { execution_order.Executed(2); });
  time_controller->AdvanceTime(TimeDelta::Millis(100));
  EXPECT_THAT(execution_order.order(), ElementsAreArray({1, 2}));
  // Destroy `thread` before `execution_order` to be sure `execution_order`
  // is not accessed on the posted task after it is destroyed.
  thread = nullptr;
}

TEST_P(SimulatedRealTimeControllerConformanceTest, ThreadPostDelayedOrderTest) {
  std::unique_ptr<TimeController> time_controller =
      CreateTimeController(GetParam());
  std::unique_ptr<rtc::Thread> thread = time_controller->CreateThread("thread");

  ExecutionOrderKeeper execution_order;
  thread->PostDelayedTask([&]() { execution_order.Executed(2); },
                          TimeDelta::Millis(500));
  thread->PostTask([&]() { execution_order.Executed(1); });
  time_controller->AdvanceTime(TimeDelta::Millis(600));
  EXPECT_THAT(execution_order.order(), ElementsAreArray({1, 2}));
  // Destroy `thread` before `execution_order` to be sure `execution_order`
  // is not accessed on the posted task after it is destroyed.
  thread = nullptr;
}

TEST_P(SimulatedRealTimeControllerConformanceTest, ThreadPostInvokeOrderTest) {
  std::unique_ptr<TimeController> time_controller =
      CreateTimeController(GetParam());
  std::unique_ptr<rtc::Thread> thread = time_controller->CreateThread("thread");

  // Tasks on thread have to be executed in order in which they were
  // posted/invoked.
  ExecutionOrderKeeper execution_order;
  thread->PostTask([&]() { execution_order.Executed(1); });
  thread->BlockingCall([&]() { execution_order.Executed(2); });
  time_controller->AdvanceTime(TimeDelta::Millis(100));
  EXPECT_THAT(execution_order.order(), ElementsAreArray({1, 2}));
  // Destroy `thread` before `execution_order` to be sure `execution_order`
  // is not accessed on the posted task after it is destroyed.
  thread = nullptr;
}

TEST_P(SimulatedRealTimeControllerConformanceTest,
       ThreadPostInvokeFromThreadOrderTest) {
  std::unique_ptr<TimeController> time_controller =
      CreateTimeController(GetParam());
  std::unique_ptr<rtc::Thread> thread = time_controller->CreateThread("thread");

  // If task is invoked from thread X on thread X it has to be executed
  // immediately.
  ExecutionOrderKeeper execution_order;
  thread->PostTask([&]() {
    thread->PostTask([&]() { execution_order.Executed(2); });
    thread->BlockingCall([&]() { execution_order.Executed(1); });
  });
  time_controller->AdvanceTime(TimeDelta::Millis(100));
  EXPECT_THAT(execution_order.order(), ElementsAreArray({1, 2}));
  // Destroy `thread` before `execution_order` to be sure `execution_order`
  // is not accessed on the posted task after it is destroyed.
  thread = nullptr;
}

TEST_P(SimulatedRealTimeControllerConformanceTest,
       TaskQueuePostEventWaitOrderTest) {
  std::unique_ptr<TimeController> time_controller =
      CreateTimeController(GetParam());
  auto task_queue = time_controller->GetTaskQueueFactory()->CreateTaskQueue(
      "task_queue", webrtc::TaskQueueFactory::Priority::NORMAL);

  // Tasks on thread have to be executed in order in which they were
  // posted/invoked.
  ExecutionOrderKeeper execution_order;
  rtc::Event event;
  task_queue->PostTask([&]() { execution_order.Executed(1); });
  task_queue->PostTask([&]() {
    execution_order.Executed(2);
    event.Set();
  });
  EXPECT_TRUE(event.Wait(/*give_up_after=*/TimeDelta::Millis(100)));
  time_controller->AdvanceTime(TimeDelta::Millis(100));
  EXPECT_THAT(execution_order.order(), ElementsAreArray({1, 2}));
  // Destroy `task_queue` before `execution_order` to be sure `execution_order`
  // is not accessed on the posted task after it is destroyed.
  task_queue = nullptr;
}

INSTANTIATE_TEST_SUITE_P(ConformanceTest,
                         SimulatedRealTimeControllerConformanceTest,
                         Values(TimeMode::kRealTime, TimeMode::kSimulated),
                         ParamsToString);

}  // namespace
}  // namespace webrtc
