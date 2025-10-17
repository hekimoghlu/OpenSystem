/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
#include "video/corruption_detection/frame_pair_corruption_score.h"

#include <optional>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/scoped_refptr.h"
#include "api/video/i420_buffer.h"
#include "api/video/video_frame_buffer.h"
#include "rtc_base/checks.h"
#include "video/corruption_detection/generic_mapping_functions.h"
#include "video/corruption_detection/halton_frame_sampler.h"
#include "video/corruption_detection/utils.h"

namespace webrtc {
namespace {

constexpr float kDefaultSampleFraction = 0.5;

}

FramePairCorruptionScorer::FramePairCorruptionScorer(
    absl::string_view codec_name,
    float scale_factor,
    std::optional<float> sample_fraction)
    : codec_type_(GetVideoCodecType(codec_name)),
      sample_fraction_(sample_fraction.value_or(kDefaultSampleFraction)),
      corruption_classifier_(scale_factor) {
  RTC_CHECK_GE(sample_fraction_, 0) << "Sample fraction must be non-negative.";
  RTC_CHECK_LE(sample_fraction_, 1) << "Sample fraction must be less than or "
                                       "equal to 1.";
}

FramePairCorruptionScorer::FramePairCorruptionScorer(
    absl::string_view codec_name,
    float growth_rate,
    float midpoint,
    std::optional<float> sample_fraction)
    : codec_type_(GetVideoCodecType(codec_name)),
      sample_fraction_(sample_fraction.value_or(kDefaultSampleFraction)),
      corruption_classifier_(growth_rate, midpoint) {
  RTC_CHECK_GE(sample_fraction_, 0) << "Sample fraction must be non-negative.";
  RTC_CHECK_LE(sample_fraction_, 1) << "Sample fraction must be less than or "
                                       "equal to 1.";
}

double FramePairCorruptionScorer::CalculateScore(
    int qp,
    I420BufferInterface& reference_buffer,
    I420BufferInterface& test_buffer) {
  RTC_CHECK_GE(reference_buffer.width(), test_buffer.width());
  RTC_CHECK_GE(reference_buffer.height(), test_buffer.height());
  // Adapted for VP9 and AV1.
  RTC_DCHECK_GE(qp, 0);
  RTC_DCHECK_LE(qp, 255);

  // We calculate corruption score per "sample" rather than per "pixel", hence
  // times "3/2".
  const int num_samples = static_cast<int>(
      (test_buffer.width() * test_buffer.height() * 3 / 2) * sample_fraction_);
  std::vector<HaltonFrameSampler::Coordinates> halton_samples =
      halton_frame_sampler_.GetSampleCoordinatesForFrame(num_samples);
  RTC_DCHECK_EQ(halton_samples.size(), num_samples);

  scoped_refptr<I420Buffer> reference_i420_buffer =
      GetAsI420Buffer(reference_buffer.ToI420());
  scoped_refptr<I420Buffer> test_i420_buffer =
      GetAsI420Buffer(test_buffer.ToI420());

  FilterSettings filter_settings = GetCorruptionFilterSettings(qp, codec_type_);

  const std::vector<FilteredSample> filtered_reference_sample_values =
      GetSampleValuesForFrame(
          reference_i420_buffer, halton_samples, test_i420_buffer->width(),
          test_i420_buffer->height(), filter_settings.std_dev);
  const std::vector<FilteredSample> filtered_test_sample_values =
      GetSampleValuesForFrame(
          test_i420_buffer, halton_samples, test_i420_buffer->width(),
          test_i420_buffer->height(), filter_settings.std_dev);
  RTC_CHECK_EQ(filtered_reference_sample_values.size(),
               filtered_test_sample_values.size());

  return corruption_classifier_.CalculateCorruptionProbability(
      filtered_reference_sample_values, filtered_test_sample_values,
      filter_settings.luma_error_threshold,
      filter_settings.chroma_error_threshold);
}

}  // namespace webrtc
