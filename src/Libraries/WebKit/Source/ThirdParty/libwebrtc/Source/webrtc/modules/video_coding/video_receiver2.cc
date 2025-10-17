/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
#include "modules/video_coding/video_receiver2.h"

#include <stddef.h>

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "api/video_codecs/video_codec.h"
#include "api/video_codecs/video_decoder.h"
#include "common_video/include/corruption_score_calculator.h"
#include "modules/video_coding/decoder_database.h"
#include "modules/video_coding/generic_decoder.h"
#include "modules/video_coding/include/video_coding_defines.h"
#include "modules/video_coding/timing/timing.h"
#include "rtc_base/checks.h"
#include "rtc_base/trace_event.h"
#include "system_wrappers/include/clock.h"

namespace webrtc {

VideoReceiver2::VideoReceiver2(
    Clock* clock,
    VCMTiming* timing,
    const FieldTrialsView& field_trials,
    CorruptionScoreCalculator* corruption_score_calculator)
    : clock_(clock),
      decoded_frame_callback_(timing,
                              clock_,
                              field_trials,
                              corruption_score_calculator),
      codec_database_() {
  decoder_sequence_checker_.Detach();
}

VideoReceiver2::~VideoReceiver2() {
  RTC_DCHECK_RUN_ON(&construction_sequence_checker_);
}

// Register a receive callback. Will be called whenever there is a new frame
// ready for rendering.
int32_t VideoReceiver2::RegisterReceiveCallback(
    VCMReceiveCallback* receive_callback) {
  RTC_DCHECK_RUN_ON(&construction_sequence_checker_);
  // This value is set before the decoder thread starts and unset after
  // the decoder thread has been stopped.
  decoded_frame_callback_.SetUserReceiveCallback(receive_callback);
  return VCM_OK;
}

void VideoReceiver2::RegisterExternalDecoder(
    std::unique_ptr<VideoDecoder> decoder,
    uint8_t payload_type) {
  RTC_DCHECK_RUN_ON(&decoder_sequence_checker_);
  RTC_DCHECK(decoded_frame_callback_.UserReceiveCallback());

  if (decoder) {
    RTC_DCHECK(!codec_database_.IsExternalDecoderRegistered(payload_type));
    codec_database_.RegisterExternalDecoder(payload_type, std::move(decoder));
  } else {
    codec_database_.DeregisterExternalDecoder(payload_type);
  }
}

bool VideoReceiver2::IsExternalDecoderRegistered(uint8_t payload_type) const {
  RTC_DCHECK_RUN_ON(&decoder_sequence_checker_);
  return codec_database_.IsExternalDecoderRegistered(payload_type);
}

// Must be called from inside the receive side critical section.
int32_t VideoReceiver2::Decode(const EncodedFrame* frame) {
  RTC_DCHECK_RUN_ON(&decoder_sequence_checker_);
  TRACE_EVENT0("webrtc", "VideoReceiver2::Decode");
  // Change decoder if payload type has changed.
  VCMGenericDecoder* decoder =
      codec_database_.GetDecoder(*frame, &decoded_frame_callback_);
  if (decoder == nullptr) {
    return VCM_NO_CODEC_REGISTERED;
  }
  return decoder->Decode(*frame, clock_->CurrentTime());
}

// Register possible receive codecs, can be called multiple times.
// Called before decoder thread is started.
void VideoReceiver2::RegisterReceiveCodec(
    uint8_t payload_type,
    const VideoDecoder::Settings& settings) {
  RTC_DCHECK_RUN_ON(&construction_sequence_checker_);
  codec_database_.RegisterReceiveCodec(payload_type, settings);
}

void VideoReceiver2::DeregisterReceiveCodec(uint8_t payload_type) {
  RTC_DCHECK_RUN_ON(&construction_sequence_checker_);
  codec_database_.DeregisterReceiveCodec(payload_type);
}

void VideoReceiver2::DeregisterReceiveCodecs() {
  RTC_DCHECK_RUN_ON(&construction_sequence_checker_);
  codec_database_.DeregisterReceiveCodecs();
}

}  // namespace webrtc
