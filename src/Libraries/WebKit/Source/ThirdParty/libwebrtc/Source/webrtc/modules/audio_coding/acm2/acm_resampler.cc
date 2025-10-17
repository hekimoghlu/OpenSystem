/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 17, 2025.
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
#include "modules/audio_coding/acm2/acm_resampler.h"

#include <string.h>

#include "api/audio/audio_frame.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace acm2 {

ACMResampler::ACMResampler() {}

ACMResampler::~ACMResampler() {}

int ACMResampler::Resample10Msec(const int16_t* in_audio,
                                 int in_freq_hz,
                                 int out_freq_hz,
                                 size_t num_audio_channels,
                                 size_t out_capacity_samples,
                                 int16_t* out_audio) {
  InterleavedView<const int16_t> src(
      in_audio, SampleRateToDefaultChannelSize(in_freq_hz), num_audio_channels);
  InterleavedView<int16_t> dst(out_audio,
                               SampleRateToDefaultChannelSize(out_freq_hz),
                               num_audio_channels);
  RTC_DCHECK_GE(out_capacity_samples, dst.size());
  if (in_freq_hz == out_freq_hz) {
    if (out_capacity_samples < src.data().size()) {
      RTC_DCHECK_NOTREACHED();
      return -1;
    }
    CopySamples(dst, src);
    RTC_DCHECK_EQ(dst.samples_per_channel(), src.samples_per_channel());
    return static_cast<int>(dst.samples_per_channel());
  }

  int out_length = resampler_.Resample(src, dst);
  if (out_length == -1) {
    RTC_LOG(LS_ERROR) << "Resample(" << in_audio << ", " << src.data().size()
                      << ", " << out_audio << ", " << out_capacity_samples
                      << ") failed.";
    return -1;
  }
  RTC_DCHECK_EQ(out_length, dst.size());
  RTC_DCHECK_EQ(out_length / num_audio_channels, dst.samples_per_channel());
  return static_cast<int>(dst.samples_per_channel());
}

ResamplerHelper::ResamplerHelper() {
  ClearSamples(last_audio_buffer_);
}

bool ResamplerHelper::MaybeResample(int desired_sample_rate_hz,
                                    AudioFrame* audio_frame) {
  const int current_sample_rate_hz = audio_frame->sample_rate_hz_;
  RTC_DCHECK_NE(current_sample_rate_hz, 0);

  // Update if resampling is required.
  const bool need_resampling =
      (desired_sample_rate_hz != -1) &&
      (current_sample_rate_hz != desired_sample_rate_hz);

  if (need_resampling && !resampled_last_output_frame_) {
    // Prime the resampler with the last frame.
    int16_t temp_output[AudioFrame::kMaxDataSizeSamples];
    int samples_per_channel_int = resampler_.Resample10Msec(
        last_audio_buffer_.data(), current_sample_rate_hz,
        desired_sample_rate_hz, audio_frame->num_channels_,
        AudioFrame::kMaxDataSizeSamples, temp_output);
    if (samples_per_channel_int < 0) {
      RTC_LOG(LS_ERROR) << "AcmReceiver::GetAudio - "
                           "Resampling last_audio_buffer_ failed.";
      return false;
    }
  }

  // TODO(bugs.webrtc.org/3923) Glitches in the output may appear if the output
  // rate from NetEq changes.
  if (need_resampling) {
    // TODO(yujo): handle this more efficiently for muted frames.
    int samples_per_channel_int = resampler_.Resample10Msec(
        audio_frame->data(), current_sample_rate_hz, desired_sample_rate_hz,
        audio_frame->num_channels_, AudioFrame::kMaxDataSizeSamples,
        audio_frame->mutable_data());
    if (samples_per_channel_int < 0) {
      RTC_LOG(LS_ERROR)
          << "AcmReceiver::GetAudio - Resampling audio_buffer_ failed.";
      return false;
    }
    audio_frame->samples_per_channel_ =
        static_cast<size_t>(samples_per_channel_int);
    audio_frame->sample_rate_hz_ = desired_sample_rate_hz;
    RTC_DCHECK_EQ(
        audio_frame->sample_rate_hz_,
        rtc::dchecked_cast<int>(audio_frame->samples_per_channel_ * 100));
    resampled_last_output_frame_ = true;
  } else {
    resampled_last_output_frame_ = false;
    // We might end up here ONLY if codec is changed.
  }

  // Store current audio in `last_audio_buffer_` for next time.
  // TODO: b/335805780 - Use CopySamples().
  memcpy(last_audio_buffer_.data(), audio_frame->data(),
         sizeof(int16_t) * audio_frame->samples_per_channel_ *
             audio_frame->num_channels_);

  return true;
}

}  // namespace acm2
}  // namespace webrtc
