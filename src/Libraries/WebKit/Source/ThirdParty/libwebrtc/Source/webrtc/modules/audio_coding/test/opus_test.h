/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#ifndef MODULES_AUDIO_CODING_TEST_OPUS_TEST_H_
#define MODULES_AUDIO_CODING_TEST_OPUS_TEST_H_

#include <math.h>

#include <memory>

#include "api/neteq/neteq.h"
#include "modules/audio_coding/acm2/acm_resampler.h"
#include "modules/audio_coding/codecs/opus/opus_interface.h"
#include "modules/audio_coding/test/PCMFile.h"
#include "modules/audio_coding/test/TestStereo.h"

namespace webrtc {

class OpusTest {
 public:
  OpusTest();
  ~OpusTest();

  void Perform();

 private:
  void Run(TestPackStereo* channel,
           size_t channels,
           int bitrate,
           size_t frame_length,
           int percent_loss = 0);

  void OpenOutFile(int test_number);

  std::unique_ptr<NetEq> neteq_;
  acm2::ResamplerHelper resampler_helper_;
  TestPackStereo* channel_a2b_;
  PCMFile in_file_stereo_;
  PCMFile in_file_mono_;
  PCMFile out_file_;
  PCMFile out_file_standalone_;
  int counter_;
  uint8_t payload_type_;
  uint32_t rtp_timestamp_;
  acm2::ACMResampler resampler_;
  WebRtcOpusEncInst* opus_mono_encoder_;
  WebRtcOpusEncInst* opus_stereo_encoder_;
  WebRtcOpusDecInst* opus_mono_decoder_;
  WebRtcOpusDecInst* opus_stereo_decoder_;
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_CODING_TEST_OPUS_TEST_H_
