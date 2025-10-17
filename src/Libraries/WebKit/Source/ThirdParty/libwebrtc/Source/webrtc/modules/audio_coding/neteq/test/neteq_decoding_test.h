/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 13, 2022.
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
#ifndef MODULES_AUDIO_CODING_NETEQ_TEST_NETEQ_DECODING_TEST_H_
#define MODULES_AUDIO_CODING_NETEQ_TEST_NETEQ_DECODING_TEST_H_

#include <memory>
#include <set>
#include <string>

#include "absl/strings/string_view.h"
#include "api/audio/audio_frame.h"
#include "api/environment/environment.h"
#include "api/neteq/neteq.h"
#include "api/rtp_headers.h"
#include "modules/audio_coding/neteq/tools/packet.h"
#include "modules/audio_coding/neteq/tools/rtp_file_source.h"
#include "system_wrappers/include/clock.h"
#include "test/gtest.h"

namespace webrtc {

class NetEqDecodingTest : public ::testing::Test {
 protected:
  // NetEQ must be polled for data once every 10 ms.
  // Thus, none of the constants below can be changed.
  static constexpr int kTimeStepMs = 10;
  static constexpr size_t kBlockSize8kHz = kTimeStepMs * 8;
  static constexpr size_t kBlockSize16kHz = kTimeStepMs * 16;
  static constexpr size_t kBlockSize32kHz = kTimeStepMs * 32;
  static constexpr size_t kBlockSize48kHz = kTimeStepMs * 48;
  static constexpr int kInitSampleRateHz = 8000;

  NetEqDecodingTest();
  virtual void SetUp();
  virtual void TearDown();
  void OpenInputFile(absl::string_view rtp_file);
  void Process();

  void DecodeAndCompare(absl::string_view rtp_file,
                        absl::string_view output_checksum,
                        absl::string_view network_stats_checksum,
                        bool gen_ref);

  static void PopulateRtpInfo(int frame_index,
                              int timestamp,
                              RTPHeader* rtp_info);
  static void PopulateCng(int frame_index,
                          int timestamp,
                          RTPHeader* rtp_info,
                          uint8_t* payload,
                          size_t* payload_len);

  void WrapTest(uint16_t start_seq_no,
                uint32_t start_timestamp,
                const std::set<uint16_t>& drop_seq_numbers,
                bool expect_seq_no_wrap,
                bool expect_timestamp_wrap);

  void LongCngWithClockDrift(double drift_factor,
                             double network_freeze_ms,
                             bool pull_audio_during_freeze,
                             int delay_tolerance_ms,
                             int max_time_to_speech_ms);

  SimulatedClock clock_;
  const Environment env_;
  std::unique_ptr<NetEq> neteq_;
  NetEq::Config config_;
  std::unique_ptr<test::RtpFileSource> rtp_source_;
  std::unique_ptr<test::Packet> packet_;
  AudioFrame out_frame_;
  int output_sample_rate_;
  int algorithmic_delay_ms_;
};

class NetEqDecodingTestTwoInstances : public NetEqDecodingTest {
 public:
  NetEqDecodingTestTwoInstances() : NetEqDecodingTest() {}

  void SetUp() override;

  void CreateSecondInstance();

 protected:
  std::unique_ptr<NetEq> neteq2_;
  NetEq::Config config2_;
};

}  // namespace webrtc
#endif  // MODULES_AUDIO_CODING_NETEQ_TEST_NETEQ_DECODING_TEST_H_
