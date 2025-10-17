/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 6, 2025.
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
#ifndef MODULES_AUDIO_CODING_TEST_PACKETLOSSTEST_H_
#define MODULES_AUDIO_CODING_TEST_PACKETLOSSTEST_H_

#include <string>

#include "absl/strings/string_view.h"
#include "modules/audio_coding/test/EncodeDecodeTest.h"

namespace webrtc {

class ReceiverWithPacketLoss : public Receiver {
 public:
  ReceiverWithPacketLoss();
  void Setup(NetEq* neteq,
             RTPStream* rtpStream,
             absl::string_view out_file_name,
             int channels,
             int file_num,
             int loss_rate,
             int burst_length);
  bool IncomingPacket() override;

 protected:
  bool PacketLost();
  int loss_rate_;
  int burst_length_;
  int packet_counter_;
  int lost_packet_counter_;
  int burst_lost_counter_;
};

class SenderWithFEC : public Sender {
 public:
  SenderWithFEC();
  void Setup(const Environment& env,
             AudioCodingModule* acm,
             RTPStream* rtpStream,
             absl::string_view in_file_name,
             int payload_type,
             SdpAudioFormat format,
             int expected_loss_rate);
  bool SetPacketLossRate(int expected_loss_rate);
  bool SetFEC(bool enable_fec);

 protected:
  int expected_loss_rate_;
};

class PacketLossTest {
 public:
  PacketLossTest(int channels,
                 int expected_loss_rate_,
                 int actual_loss_rate,
                 int burst_length);
  void Perform();

 protected:
  int channels_;
  std::string in_file_name_;
  int sample_rate_hz_;
  int expected_loss_rate_;
  int actual_loss_rate_;
  int burst_length_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_TEST_PACKETLOSSTEST_H_
