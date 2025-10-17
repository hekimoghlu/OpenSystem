/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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
#ifndef RTC_TOOLS_NETWORK_TESTER_TEST_CONTROLLER_H_
#define RTC_TOOLS_NETWORK_TESTER_TEST_CONTROLLER_H_

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <memory>
#include <optional>
#include <string>

#include "api/sequence_checker.h"
#include "p2p/base/basic_packet_socket_factory.h"
#include "rtc_base/async_packet_socket.h"
#include "rtc_base/network/received_packet.h"
#include "rtc_base/socket_address.h"
#include "rtc_base/socket_server.h"
#include "rtc_base/synchronization/mutex.h"
#include "rtc_base/system/no_unique_address.h"
#include "rtc_base/thread.h"
#include "rtc_base/thread_annotations.h"
#include "rtc_tools/network_tester/packet_logger.h"
#include "rtc_tools/network_tester/packet_sender.h"

#ifdef WEBRTC_NETWORK_TESTER_PROTO
#include "rtc_tools/network_tester/network_tester_packet.pb.h"
using webrtc::network_tester::packet::NetworkTesterPacket;
#else
class NetworkTesterPacket;
#endif  // WEBRTC_NETWORK_TESTER_PROTO

namespace webrtc {

constexpr size_t kEthernetMtu = 1500;

class TestController {
 public:
  TestController(int min_port,
                 int max_port,
                 const std::string& config_file_path,
                 const std::string& log_file_path);
  ~TestController();

  TestController(const TestController&) = delete;
  TestController& operator=(const TestController&) = delete;

  void SendConnectTo(const std::string& hostname, int port);

  void SendData(const NetworkTesterPacket& packet,
                std::optional<size_t> data_size);

  void OnTestDone();

  bool IsTestDone();

 private:
  void OnReadPacket(rtc::AsyncPacketSocket* socket,
                    const rtc::ReceivedPacket& received_packet);
  RTC_NO_UNIQUE_ADDRESS SequenceChecker test_controller_thread_checker_;
  std::unique_ptr<rtc::SocketServer> socket_server_;
  std::unique_ptr<rtc::Thread> packet_sender_thread_;
  rtc::BasicPacketSocketFactory socket_factory_
      RTC_GUARDED_BY(packet_sender_thread_);
  const std::string config_file_path_;
  PacketLogger packet_logger_ RTC_GUARDED_BY(packet_sender_thread_);
  Mutex test_done_lock_ RTC_GUARDED_BY(test_controller_thread_checker_);
  bool local_test_done_ RTC_GUARDED_BY(test_done_lock_);
  bool remote_test_done_ RTC_GUARDED_BY(test_done_lock_);
  std::array<char, kEthernetMtu> send_data_
      RTC_GUARDED_BY(packet_sender_thread_);
  std::unique_ptr<rtc::AsyncPacketSocket> udp_socket_
      RTC_GUARDED_BY(packet_sender_thread_);
  rtc::SocketAddress remote_address_;
  std::unique_ptr<PacketSender> packet_sender_
      RTC_GUARDED_BY(packet_sender_thread_);
  rtc::scoped_refptr<webrtc::PendingTaskSafetyFlag> task_safety_flag_;
};

}  // namespace webrtc

#endif  // RTC_TOOLS_NETWORK_TESTER_TEST_CONTROLLER_H_
