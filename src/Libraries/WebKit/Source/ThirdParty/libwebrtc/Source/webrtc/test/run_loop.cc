/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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

#include "rtc_base/time_utils.h"

namespace webrtc {
namespace test {

RunLoop::RunLoop() {
  worker_thread_.WrapCurrent();
}

RunLoop::~RunLoop() {
  worker_thread_.UnwrapCurrent();
}

TaskQueueBase* RunLoop::task_queue() {
  return &worker_thread_;
}

void RunLoop::Run() {
  worker_thread_.ProcessMessages(WorkerThread::kForever);
}

void RunLoop::Quit() {
  socket_server_.FailNextWait();
}

void RunLoop::Flush() {
  worker_thread_.PostTask([this]() { socket_server_.FailNextWait(); });
  // If a test clock is used, like with GlobalSimulatedTimeController then the
  // thread will loop forever since time never increases. Since the clock is
  // simulated, 0ms can be used as the loop delay, which will process all
  // messages ready for execution.
  int cms = rtc::GetClockForTesting() ? 0 : 1000;
  worker_thread_.ProcessMessages(cms);
}

RunLoop::FakeSocketServer::FakeSocketServer() = default;
RunLoop::FakeSocketServer::~FakeSocketServer() = default;

void RunLoop::FakeSocketServer::FailNextWait() {
  fail_next_wait_ = true;
}

bool RunLoop::FakeSocketServer::Wait(webrtc::TimeDelta max_wait_duration,
                                     bool process_io) {
  if (fail_next_wait_) {
    fail_next_wait_ = false;
    return false;
  }
  return true;
}

void RunLoop::FakeSocketServer::WakeUp() {}

rtc::Socket* RunLoop::FakeSocketServer::CreateSocket(int family, int type) {
  return nullptr;
}

RunLoop::WorkerThread::WorkerThread(rtc::SocketServer* ss)
    : rtc::Thread(ss), tq_setter_(this) {}

}  // namespace test
}  // namespace webrtc
