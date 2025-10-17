/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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
#include "modules/audio_processing/include/audio_frame_proxies.h"

#include "api/audio/audio_frame.h"
#include "api/audio/audio_processing.h"

namespace webrtc {

int ProcessAudioFrame(AudioProcessing* ap, AudioFrame* frame) {
  if (!frame || !ap) {
    return AudioProcessing::Error::kNullPointerError;
  }

  StreamConfig input_config(frame->sample_rate_hz_, frame->num_channels_);
  StreamConfig output_config(frame->sample_rate_hz_, frame->num_channels_);
  RTC_DCHECK_EQ(frame->samples_per_channel(), input_config.num_frames());

  int result = ap->ProcessStream(frame->data(), input_config, output_config,
                                 frame->mutable_data());

  AudioProcessingStats stats = ap->GetStatistics();

  if (stats.voice_detected) {
    frame->vad_activity_ = *stats.voice_detected
                               ? AudioFrame::VADActivity::kVadActive
                               : AudioFrame::VADActivity::kVadPassive;
  }

  return result;
}

int ProcessReverseAudioFrame(AudioProcessing* ap, AudioFrame* frame) {
  if (!frame || !ap) {
    return AudioProcessing::Error::kNullPointerError;
  }

  // Must be a native rate.
  if (frame->sample_rate_hz_ != AudioProcessing::NativeRate::kSampleRate8kHz &&
      frame->sample_rate_hz_ != AudioProcessing::NativeRate::kSampleRate16kHz &&
      frame->sample_rate_hz_ != AudioProcessing::NativeRate::kSampleRate32kHz &&
      frame->sample_rate_hz_ != AudioProcessing::NativeRate::kSampleRate48kHz) {
    return AudioProcessing::Error::kBadSampleRateError;
  }

  if (frame->num_channels_ <= 0) {
    return AudioProcessing::Error::kBadNumberChannelsError;
  }

  StreamConfig input_config(frame->sample_rate_hz_, frame->num_channels_);
  StreamConfig output_config(frame->sample_rate_hz_, frame->num_channels_);

  int result = ap->ProcessReverseStream(frame->data(), input_config,
                                        output_config, frame->mutable_data());
  return result;
}

}  // namespace webrtc
