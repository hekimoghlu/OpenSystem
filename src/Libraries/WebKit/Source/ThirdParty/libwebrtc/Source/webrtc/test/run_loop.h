/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 18, 2023.
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
#ifndef TEST_RUN_LOOP_H_
#define TEST_RUN_LOOP_H_

#include <utility>

#include "absl/functional/any_invocable.h"
#include "api/task_queue/task_queue_base.h"
#include "rtc_base/thread.h"

namespace webrtc {
namespace test {

// This utility class allows you to run a TaskQueue supported interface on the
// main test thread, call Run() while doing things asynchonously and break
// the loop (from the same thread) from a callback by calling Quit().
class RunLoop {
 public:
  RunLoop();
  ~RunLoop();

  TaskQueueBase* task_queue();

  void Run();
  void Quit();

  void Flush();

  void PostTask(absl::AnyInvocable<void() &&> task) {
    task_queue()->PostTask(std::move(task));
  }

 private:
  class FakeSocketServer : public rtc::SocketServer {
   public:
    FakeSocketServer();
    ~FakeSocketServer();

    void FailNextWait();

   private:
    bool Wait(webrtc::TimeDelta max_wait_duration, bool process_io) override;
    void WakeUp() override;

    rtc::Socket* CreateSocket(int family, int type) override;

   private:
    bool fail_next_wait_ = false;
  };

  class WorkerThread : public rtc::Thread {
   public:
    explicit WorkerThread(rtc::SocketServer* ss);

   private:
    CurrentTaskQueueSetter tq_setter_;
  };

  FakeSocketServer socket_server_;
  WorkerThread worker_thread_{&socket_server_};
};

}  // namespace test
}  // namespace webrtc

#endif  // TEST_RUN_LOOP_H_
