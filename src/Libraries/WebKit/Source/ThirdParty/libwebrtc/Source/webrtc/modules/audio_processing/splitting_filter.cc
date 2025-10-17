/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 27, 2022.
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
#include "modules/audio_processing/splitting_filter.h"

#include <array>

#include "api/array_view.h"
#include "common_audio/channel_buffer.h"
#include "common_audio/signal_processing/include/signal_processing_library.h"
#include "rtc_base/checks.h"

namespace webrtc {
namespace {

constexpr size_t kSamplesPerBand = 160;
constexpr size_t kTwoBandFilterSamplesPerFrame = 320;

}  // namespace

SplittingFilter::SplittingFilter(size_t num_channels,
                                 size_t num_bands,
                                 size_t /* num_frames */)
    : num_bands_(num_bands),
      two_bands_states_(num_bands_ == 2 ? num_channels : 0),
      three_band_filter_banks_(num_bands_ == 3 ? num_channels : 0) {
  RTC_CHECK(num_bands_ == 2 || num_bands_ == 3);
}

SplittingFilter::~SplittingFilter() = default;

void SplittingFilter::Analysis(const ChannelBuffer<float>* data,
                               ChannelBuffer<float>* bands) {
  RTC_DCHECK_EQ(num_bands_, bands->num_bands());
  RTC_DCHECK_EQ(data->num_channels(), bands->num_channels());
  RTC_DCHECK_EQ(data->num_frames(),
                bands->num_frames_per_band() * bands->num_bands());
  if (bands->num_bands() == 2) {
    TwoBandsAnalysis(data, bands);
  } else if (bands->num_bands() == 3) {
    ThreeBandsAnalysis(data, bands);
  }
}

void SplittingFilter::Synthesis(const ChannelBuffer<float>* bands,
                                ChannelBuffer<float>* data) {
  RTC_DCHECK_EQ(num_bands_, bands->num_bands());
  RTC_DCHECK_EQ(data->num_channels(), bands->num_channels());
  RTC_DCHECK_EQ(data->num_frames(),
                bands->num_frames_per_band() * bands->num_bands());
  if (bands->num_bands() == 2) {
    TwoBandsSynthesis(bands, data);
  } else if (bands->num_bands() == 3) {
    ThreeBandsSynthesis(bands, data);
  }
}

void SplittingFilter::TwoBandsAnalysis(const ChannelBuffer<float>* data,
                                       ChannelBuffer<float>* bands) {
  RTC_DCHECK_EQ(two_bands_states_.size(), data->num_channels());
  RTC_DCHECK_EQ(data->num_frames(), kTwoBandFilterSamplesPerFrame);

  for (size_t i = 0; i < two_bands_states_.size(); ++i) {
    std::array<std::array<int16_t, kSamplesPerBand>, 2> bands16;
    std::array<int16_t, kTwoBandFilterSamplesPerFrame> full_band16;
    FloatS16ToS16(data->channels(0)[i], full_band16.size(), full_band16.data());
    WebRtcSpl_AnalysisQMF(full_band16.data(), data->num_frames(),
                          bands16[0].data(), bands16[1].data(),
                          two_bands_states_[i].analysis_state1,
                          two_bands_states_[i].analysis_state2);
    S16ToFloatS16(bands16[0].data(), bands16[0].size(), bands->channels(0)[i]);
    S16ToFloatS16(bands16[1].data(), bands16[1].size(), bands->channels(1)[i]);
  }
}

void SplittingFilter::TwoBandsSynthesis(const ChannelBuffer<float>* bands,
                                        ChannelBuffer<float>* data) {
  RTC_DCHECK_LE(data->num_channels(), two_bands_states_.size());
  RTC_DCHECK_EQ(data->num_frames(), kTwoBandFilterSamplesPerFrame);
  for (size_t i = 0; i < data->num_channels(); ++i) {
    std::array<std::array<int16_t, kSamplesPerBand>, 2> bands16;
    std::array<int16_t, kTwoBandFilterSamplesPerFrame> full_band16;
    FloatS16ToS16(bands->channels(0)[i], bands16[0].size(), bands16[0].data());
    FloatS16ToS16(bands->channels(1)[i], bands16[1].size(), bands16[1].data());
    WebRtcSpl_SynthesisQMF(bands16[0].data(), bands16[1].data(),
                           bands->num_frames_per_band(), full_band16.data(),
                           two_bands_states_[i].synthesis_state1,
                           two_bands_states_[i].synthesis_state2);
    S16ToFloatS16(full_band16.data(), full_band16.size(), data->channels(0)[i]);
  }
}

void SplittingFilter::ThreeBandsAnalysis(const ChannelBuffer<float>* data,
                                         ChannelBuffer<float>* bands) {
  RTC_DCHECK_EQ(three_band_filter_banks_.size(), data->num_channels());
  RTC_DCHECK_LE(data->num_channels(), three_band_filter_banks_.size());
  RTC_DCHECK_LE(data->num_channels(), bands->num_channels());
  RTC_DCHECK_EQ(data->num_frames(), ThreeBandFilterBank::kFullBandSize);
  RTC_DCHECK_EQ(bands->num_frames(), ThreeBandFilterBank::kFullBandSize);
  RTC_DCHECK_EQ(bands->num_bands(), ThreeBandFilterBank::kNumBands);
  RTC_DCHECK_EQ(bands->num_frames_per_band(),
                ThreeBandFilterBank::kSplitBandSize);

  for (size_t i = 0; i < three_band_filter_banks_.size(); ++i) {
    three_band_filter_banks_[i].Analysis(
        rtc::ArrayView<const float, ThreeBandFilterBank::kFullBandSize>(
            data->channels_view()[i].data(),
            ThreeBandFilterBank::kFullBandSize),
        rtc::ArrayView<const rtc::ArrayView<float>,
                       ThreeBandFilterBank::kNumBands>(
            bands->bands_view(i).data(), ThreeBandFilterBank::kNumBands));
  }
}

void SplittingFilter::ThreeBandsSynthesis(const ChannelBuffer<float>* bands,
                                          ChannelBuffer<float>* data) {
  RTC_DCHECK_LE(data->num_channels(), three_band_filter_banks_.size());
  RTC_DCHECK_LE(data->num_channels(), bands->num_channels());
  RTC_DCHECK_LE(data->num_channels(), three_band_filter_banks_.size());
  RTC_DCHECK_EQ(data->num_frames(), ThreeBandFilterBank::kFullBandSize);
  RTC_DCHECK_EQ(bands->num_frames(), ThreeBandFilterBank::kFullBandSize);
  RTC_DCHECK_EQ(bands->num_bands(), ThreeBandFilterBank::kNumBands);
  RTC_DCHECK_EQ(bands->num_frames_per_band(),
                ThreeBandFilterBank::kSplitBandSize);

  for (size_t i = 0; i < data->num_channels(); ++i) {
    three_band_filter_banks_[i].Synthesis(
        rtc::ArrayView<const rtc::ArrayView<float>,
                       ThreeBandFilterBank::kNumBands>(
            bands->bands_view(i).data(), ThreeBandFilterBank::kNumBands),
        rtc::ArrayView<float, ThreeBandFilterBank::kFullBandSize>(
            data->channels_view()[i].data(),
            ThreeBandFilterBank::kFullBandSize));
  }
}

}  // namespace webrtc
