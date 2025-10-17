/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 12, 2022.
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
#ifndef MODULES_AUDIO_CODING_TEST_TESTSTEREO_H_
#define MODULES_AUDIO_CODING_TEST_TESTSTEREO_H_

#include <math.h>

#include <memory>

#include "api/environment/environment.h"
#include "api/neteq/neteq.h"
#include "modules/audio_coding/acm2/acm_resampler.h"
#include "modules/audio_coding/include/audio_coding_module.h"
#include "modules/audio_coding/test/PCMFile.h"

#define PCMA_AND_PCMU

namespace webrtc {

enum StereoMonoMode { kNotSet, kMono, kStereo };

class TestPackStereo : public AudioPacketizationCallback {
 public:
  TestPackStereo();
  ~TestPackStereo();

  void RegisterReceiverNetEq(NetEq* neteq);

  int32_t SendData(AudioFrameType frame_type,
                   uint8_t payload_type,
                   uint32_t timestamp,
                   const uint8_t* payload_data,
                   size_t payload_size,
                   int64_t absolute_capture_timestamp_ms) override;

  uint16_t payload_size();
  uint32_t timestamp_diff();
  void reset_payload_size();
  void set_codec_mode(StereoMonoMode mode);
  void set_lost_packet(bool lost);

 private:
  NetEq* neteq_;
  int16_t seq_no_;
  uint32_t timestamp_diff_;
  uint32_t last_in_timestamp_;
  uint64_t total_bytes_;
  int payload_size_;
  StereoMonoMode codec_mode_;
  // Simulate packet losses
  bool lost_packet_;
};

class TestStereo {
 public:
  TestStereo();
  ~TestStereo();

  void Perform();

 private:
  // The default value of '-1' indicates that the registration is based only on
  // codec name and a sampling frequncy matching is not required. This is useful
  // for codecs which support several sampling frequency.
  void RegisterSendCodec(char side,
                         char* codec_name,
                         int32_t samp_freq_hz,
                         int rate,
                         int pack_size,
                         int channels);

  void Run(TestPackStereo* channel,
           int in_channels,
           int out_channels,
           int percent_loss = 0);
  void OpenOutFile(int16_t test_number);

  const Environment env_;
  std::unique_ptr<AudioCodingModule> acm_a_;
  std::unique_ptr<NetEq> neteq_;
  acm2::ResamplerHelper resampler_helper_;

  TestPackStereo* channel_a2b_;

  PCMFile* in_file_stereo_;
  PCMFile* in_file_mono_;
  PCMFile out_file_;
  int16_t test_cntr_;
  uint16_t pack_size_samp_;
  uint16_t pack_size_bytes_;
  int counter_;
  char* send_codec_name_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_TEST_TESTSTEREO_H_
