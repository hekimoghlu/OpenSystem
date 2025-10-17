/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 15, 2024.
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
#ifndef MODULES_VIDEO_CODING_VIDEO_RECEIVER2_H_
#define MODULES_VIDEO_CODING_VIDEO_RECEIVER2_H_

#include <memory>
#include <vector>

#include "api/field_trials_view.h"
#include "api/sequence_checker.h"
#include "api/video/encoded_frame.h"
#include "api/video_codecs/video_decoder.h"
#include "common_video/include/corruption_score_calculator.h"
#include "modules/video_coding/decoder_database.h"
#include "modules/video_coding/generic_decoder.h"
#include "modules/video_coding/timing/timing.h"
#include "rtc_base/system/no_unique_address.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

// This class is a copy of vcm::VideoReceiver, trimmed down to what's used by
// VideoReceive stream, with the aim to incrementally trim it down further and
// ultimately delete it. It's difficult to do this incrementally with the
// original VideoReceiver class, since it is used by the legacy
// VideoCodingModule api.
class VideoReceiver2 {
 public:
  VideoReceiver2(Clock* clock,
                 VCMTiming* timing,
                 const FieldTrialsView& field_trials,
                 CorruptionScoreCalculator* corruption_score_calculator);
  ~VideoReceiver2();

  void RegisterReceiveCodec(uint8_t payload_type,
                            const VideoDecoder::Settings& decoder_settings);
  void DeregisterReceiveCodec(uint8_t payload_type);
  void DeregisterReceiveCodecs();

  void RegisterExternalDecoder(std::unique_ptr<VideoDecoder> decoder,
                               uint8_t payload_type);

  bool IsExternalDecoderRegistered(uint8_t payload_type) const;
  int32_t RegisterReceiveCallback(VCMReceiveCallback* receive_callback);

  int32_t Decode(const EncodedFrame* frame);

 private:
  RTC_NO_UNIQUE_ADDRESS SequenceChecker construction_sequence_checker_;
  RTC_NO_UNIQUE_ADDRESS SequenceChecker decoder_sequence_checker_;
  Clock* const clock_;
  VCMDecodedFrameCallback decoded_frame_callback_;
  // Callbacks are set before the decoder thread starts.
  // Once the decoder thread has been started, usage of `_codecDataBase` moves
  // over to the decoder thread.
  VCMDecoderDatabase codec_database_;
};

}  // namespace webrtc

#endif  // MODULES_VIDEO_CODING_VIDEO_RECEIVER2_H_
