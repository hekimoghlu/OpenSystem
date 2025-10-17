/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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
#include "common_audio/resampler/include/push_resampler.h"

#include <stdint.h>
#include <string.h>

#include <memory>

#include "api/audio/audio_frame.h"
#include "common_audio/include/audio_util.h"
#include "common_audio/resampler/push_sinc_resampler.h"
#include "rtc_base/checks.h"

namespace webrtc {

namespace {
// Maximum concurrent number of channels for `PushResampler<>`.
// Note that this may be different from what the maximum is for audio codecs.
constexpr int kMaxNumberOfChannels = 8;
}  // namespace

template <typename T>
PushResampler<T>::PushResampler() = default;

template <typename T>
PushResampler<T>::PushResampler(size_t src_samples_per_channel,
                                size_t dst_samples_per_channel,
                                size_t num_channels) {
  EnsureInitialized(src_samples_per_channel, dst_samples_per_channel,
                    num_channels);
}

template <typename T>
PushResampler<T>::~PushResampler() = default;

template <typename T>
void PushResampler<T>::EnsureInitialized(size_t src_samples_per_channel,
                                         size_t dst_samples_per_channel,
                                         size_t num_channels) {
  RTC_DCHECK_GT(src_samples_per_channel, 0);
  RTC_DCHECK_GT(dst_samples_per_channel, 0);
  RTC_DCHECK_GT(num_channels, 0);
  RTC_DCHECK_LE(src_samples_per_channel, kMaxSamplesPerChannel10ms);
  RTC_DCHECK_LE(dst_samples_per_channel, kMaxSamplesPerChannel10ms);
  RTC_DCHECK_LE(num_channels, kMaxNumberOfChannels);

  if (src_samples_per_channel == SamplesPerChannel(source_view_) &&
      dst_samples_per_channel == SamplesPerChannel(destination_view_) &&
      num_channels == NumChannels(source_view_)) {
    // No-op if settings haven't changed.
    return;
  }

  // Allocate two buffers for all source and destination channels.
  // Then organize source and destination views together with an array of
  // resamplers for each channel in the deinterlaved buffers.
  source_.reset(new T[src_samples_per_channel * num_channels]);
  destination_.reset(new T[dst_samples_per_channel * num_channels]);
  source_view_ = DeinterleavedView<T>(source_.get(), src_samples_per_channel,
                                      num_channels);
  destination_view_ = DeinterleavedView<T>(
      destination_.get(), dst_samples_per_channel, num_channels);
  resamplers_.resize(num_channels);
  for (size_t i = 0; i < num_channels; ++i) {
    resamplers_[i] = std::make_unique<PushSincResampler>(
        src_samples_per_channel, dst_samples_per_channel);
  }
}

template <typename T>
int PushResampler<T>::Resample(InterleavedView<const T> src,
                               InterleavedView<T> dst) {
  EnsureInitialized(SamplesPerChannel(src), SamplesPerChannel(dst),
                    NumChannels(src));

  RTC_DCHECK_EQ(NumChannels(src), NumChannels(source_view_));
  RTC_DCHECK_EQ(NumChannels(dst), NumChannels(destination_view_));
  RTC_DCHECK_EQ(SamplesPerChannel(src), SamplesPerChannel(source_view_));
  RTC_DCHECK_EQ(SamplesPerChannel(dst), SamplesPerChannel(destination_view_));

  if (SamplesPerChannel(src) == SamplesPerChannel(dst)) {
    // The old resampler provides this memcpy facility in the case of matching
    // sample rates, so reproduce it here for the sinc resampler.
    CopySamples(dst, src);
    return static_cast<int>(src.data().size());
  }

  Deinterleave(src, source_view_);

  for (size_t i = 0; i < resamplers_.size(); ++i) {
    size_t dst_length_mono =
        resamplers_[i]->Resample(source_view_[i], destination_view_[i]);
    RTC_DCHECK_EQ(dst_length_mono, SamplesPerChannel(dst));
  }

  Interleave<T>(destination_view_, dst);
  return static_cast<int>(dst.size());
}

template <typename T>
int PushResampler<T>::Resample(MonoView<const T> src, MonoView<T> dst) {
  RTC_DCHECK_EQ(resamplers_.size(), 1);
  RTC_DCHECK_EQ(SamplesPerChannel(src), SamplesPerChannel(source_view_));
  RTC_DCHECK_EQ(SamplesPerChannel(dst), SamplesPerChannel(destination_view_));

  if (SamplesPerChannel(src) == SamplesPerChannel(dst)) {
    CopySamples(dst, src);
    return static_cast<int>(src.size());
  }

  return resamplers_[0]->Resample(src, dst);
}

// Explictly generate required instantiations.
template class PushResampler<int16_t>;
template class PushResampler<float>;

}  // namespace webrtc
