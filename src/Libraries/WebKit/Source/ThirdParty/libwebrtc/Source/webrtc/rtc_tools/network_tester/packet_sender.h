/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#ifndef RTC_TOOLS_NETWORK_TESTER_PACKET_SENDER_H_
#define RTC_TOOLS_NETWORK_TESTER_PACKET_SENDER_H_

#include <memory>
#include <string>

#include "api/sequence_checker.h"
#include "api/task_queue/pending_task_safety_flag.h"
#include "api/task_queue/task_queue_factory.h"
#include "rtc_base/system/no_unique_address.h"

#ifdef WEBRTC_NETWORK_TESTER_PROTO
#include "rtc_tools/network_tester/network_tester_packet.pb.h"
using webrtc::network_tester::packet::NetworkTesterPacket;
#else
class NetworkTesterPacket;
#endif  // WEBRTC_NETWORK_TESTER_PROTO

namespace webrtc {

class TestController;

class PacketSender {
 public:
  PacketSender(
      TestController* test_controller,
      webrtc::TaskQueueBase* worker_queue,
      rtc::scoped_refptr<webrtc::PendingTaskSafetyFlag> task_safety_flag,
      const std::string& config_file_path);
  ~PacketSender();

  PacketSender(const PacketSender&) = delete;
  PacketSender& operator=(const PacketSender&) = delete;

  void StartSending();
  void StopSending();
  bool IsSending() const;

  void SendPacket();

  int64_t GetSendIntervalMs() const;
  void UpdateTestSetting(size_t packet_size, int64_t send_interval_ms);

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker worker_queue_checker_;
  size_t packet_size_ RTC_GUARDED_BY(worker_queue_checker_);
  int64_t send_interval_ms_ RTC_GUARDED_BY(worker_queue_checker_);
  int64_t sequence_number_ RTC_GUARDED_BY(worker_queue_checker_);
  bool sending_ RTC_GUARDED_BY(worker_queue_checker_);
  const std::string config_file_path_;
  TestController* const test_controller_;
  webrtc::TaskQueueBase* worker_queue_;
  rtc::scoped_refptr<webrtc::PendingTaskSafetyFlag> task_safety_flag_;
};

}  // namespace webrtc

#endif  // RTC_TOOLS_NETWORK_TESTER_PACKET_SENDER_H_
