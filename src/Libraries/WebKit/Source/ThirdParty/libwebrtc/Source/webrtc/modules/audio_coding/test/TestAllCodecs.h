/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#ifndef MODULES_AUDIO_CODING_TEST_TESTALLCODECS_H_
#define MODULES_AUDIO_CODING_TEST_TESTALLCODECS_H_

#include <memory>

#include "api/environment/environment.h"
#include "modules/audio_coding/include/audio_coding_module.h"
#include "modules/audio_coding/test/PCMFile.h"

namespace webrtc {

class NetEq;

class TestPack : public AudioPacketizationCallback {
 public:
  TestPack();
  ~TestPack();

  void RegisterReceiverNetEq(NetEq* neteq);

  int32_t SendData(AudioFrameType frame_type,
                   uint8_t payload_type,
                   uint32_t timestamp,
                   const uint8_t* payload_data,
                   size_t payload_size,
                   int64_t absolute_capture_timestamp_ms) override;

  size_t payload_size();
  uint32_t timestamp_diff();
  void reset_payload_size();

 private:
  NetEq* neteq_;
  uint16_t sequence_number_;
  uint8_t payload_data_[60 * 32 * 2 * 2];
  uint32_t timestamp_diff_;
  uint32_t last_in_timestamp_;
  uint64_t total_bytes_;
  size_t payload_size_;
};

class TestAllCodecs {
 public:
  TestAllCodecs();
  ~TestAllCodecs();

  void Perform();

 private:
  // The default value of '-1' indicates that the registration is based only on
  // codec name, and a sampling frequency matching is not required.
  // This is useful for codecs which support several sampling frequency.
  // Note! Only mono mode is tested in this test.
  void RegisterSendCodec(char* codec_name,
                         int32_t sampling_freq_hz,
                         int rate,
                         int packet_size,
                         size_t extra_byte);

  void Run(TestPack* channel);
  void OpenOutFile(int test_number);

  const Environment env_;
  std::unique_ptr<AudioCodingModule> acm_a_;
  std::unique_ptr<NetEq> neteq_;
  TestPack* channel_a_to_b_;
  PCMFile infile_a_;
  PCMFile outfile_b_;
  int test_count_;
  int packet_size_samples_;
  size_t packet_size_bytes_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_TEST_TESTALLCODECS_H_
