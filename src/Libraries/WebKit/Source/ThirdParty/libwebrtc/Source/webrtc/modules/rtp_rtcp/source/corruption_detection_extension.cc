/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#include "modules/rtp_rtcp/source/corruption_detection_extension.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "api/array_view.h"
#include "common_video/corruption_detection_message.h"

namespace webrtc {
namespace {

constexpr size_t kMandatoryPayloadBytes = 1;
constexpr size_t kConfigurationBytes = 3;
constexpr double kMaxValueForStdDev = 40.0;

}  // namespace

// A description of the extension can be found at
// http://www.webrtc.org/experiments/rtp-hdrext/corruption-detection

bool CorruptionDetectionExtension::Parse(rtc::ArrayView<const uint8_t> data,
                                         CorruptionDetectionMessage* message) {
  if (message == nullptr) {
    return false;
  }
  if ((data.size() != kMandatoryPayloadBytes &&
       data.size() <= kConfigurationBytes) ||
      data.size() > kMaxValueSizeBytes) {
    return false;
  }
  message->interpret_sequence_index_as_most_significant_bits_ = data[0] >> 7;
  message->sequence_index_ = data[0] & 0b0111'1111;
  if (data.size() == kMandatoryPayloadBytes) {
    return true;
  }
  message->std_dev_ = data[1] * kMaxValueForStdDev / 255.0;
  uint8_t channel_error_thresholds = data[2];
  message->luma_error_threshold_ = channel_error_thresholds >> 4;
  message->chroma_error_threshold_ = channel_error_thresholds & 0xF;
  message->sample_values_.assign(data.cbegin() + kConfigurationBytes,
                                 data.cend());
  return true;
}

bool CorruptionDetectionExtension::Write(
    rtc::ArrayView<uint8_t> data,
    const CorruptionDetectionMessage& message) {
  if (data.size() != ValueSize(message) || data.size() > kMaxValueSizeBytes) {
    return false;
  }

  data[0] = message.sequence_index() & 0b0111'1111;
  if (message.interpret_sequence_index_as_most_significant_bits()) {
    data[0] |= 0b1000'0000;
  }
  if (message.sample_values().empty()) {
    return true;
  }
  data[1] = static_cast<uint8_t>(
      std::round(message.std_dev() / kMaxValueForStdDev * 255.0));
  data[2] = (message.luma_error_threshold() << 4) |
            (message.chroma_error_threshold() & 0xF);
  rtc::ArrayView<uint8_t> sample_values = data.subview(kConfigurationBytes);
  for (size_t i = 0; i < message.sample_values().size(); ++i) {
    sample_values[i] = std::floor(message.sample_values()[i]);
  }
  return true;
}

size_t CorruptionDetectionExtension::ValueSize(
    const CorruptionDetectionMessage& message) {
  if (message.sample_values_.empty()) {
    return kMandatoryPayloadBytes;
  }
  return kConfigurationBytes + message.sample_values_.size();
}

}  // namespace webrtc
