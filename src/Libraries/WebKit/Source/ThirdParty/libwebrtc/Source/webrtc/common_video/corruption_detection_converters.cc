/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 26, 2023.
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
#include "common_video/corruption_detection_converters.h"

#include <optional>
#include <vector>

#include "common_video/corruption_detection_message.h"
#include "common_video/frame_instrumentation_data.h"
#include "rtc_base/checks.h"

namespace webrtc {

namespace {

int GetFullSequenceIndex(int previous_sequence_index,
                         int sequence_index_update,
                         bool update_the_most_significant_bits) {
  RTC_CHECK_GE(previous_sequence_index, 0)
      << "previous_sequence_index must not be negative";
  RTC_CHECK_LE(previous_sequence_index, 0x7FFF)
      << "previous_sequence_index must be at most 15 bits";
  RTC_CHECK_GE(sequence_index_update, 0)
      << "sequence_index_update must not be negative";
  RTC_CHECK_LE(sequence_index_update, 0b0111'1111)
      << "sequence_index_update must be at most 7 bits";
  if (update_the_most_significant_bits) {
    // Reset LSB.
    return sequence_index_update << 7;
  }
  int upper_bits = previous_sequence_index & 0b0011'1111'1000'0000;
  if (sequence_index_update < (previous_sequence_index & 0b0111'1111)) {
    // Assume one and only one wraparound has happened.
    upper_bits += 0b1000'0000;
  }
  // Replace the lowest bits with the bits from the update.
  return upper_bits + sequence_index_update;
}

int GetSequenceIndexForMessage(int sequence_index,
                               bool communicate_upper_bits) {
  return communicate_upper_bits ? (sequence_index >> 7)
                                : (sequence_index & 0b0111'1111);
}

}  // namespace

std::optional<FrameInstrumentationData>
ConvertCorruptionDetectionMessageToFrameInstrumentationData(
    const CorruptionDetectionMessage& message,
    int previous_sequence_index) {
  if (previous_sequence_index < 0) {
    return std::nullopt;
  }
  if (message.sample_values().empty()) {
    return std::nullopt;
  }
  int full_sequence_index = GetFullSequenceIndex(
      previous_sequence_index, message.sequence_index(),
      message.interpret_sequence_index_as_most_significant_bits());
  std::vector<double> sample_values(message.sample_values().cbegin(),
                                    message.sample_values().cend());
  return FrameInstrumentationData{
      .sequence_index = full_sequence_index,
      .communicate_upper_bits =
          message.interpret_sequence_index_as_most_significant_bits(),
      .std_dev = message.std_dev(),
      .luma_error_threshold = message.luma_error_threshold(),
      .chroma_error_threshold = message.chroma_error_threshold(),
      .sample_values = sample_values};
}

std::optional<FrameInstrumentationSyncData>
ConvertCorruptionDetectionMessageToFrameInstrumentationSyncData(
    const CorruptionDetectionMessage& message,
    int previous_sequence_index) {
  if (previous_sequence_index < 0) {
    return std::nullopt;
  }
  if (!message.sample_values().empty()) {
    return std::nullopt;
  }
  if (!message.interpret_sequence_index_as_most_significant_bits()) {
    return std::nullopt;
  }
  return FrameInstrumentationSyncData{
      .sequence_index = GetFullSequenceIndex(
          previous_sequence_index, message.sequence_index(),
          /*update_the_most_significant_bits=*/true),
      .communicate_upper_bits = true};
}

std::optional<CorruptionDetectionMessage>
ConvertFrameInstrumentationDataToCorruptionDetectionMessage(
    const FrameInstrumentationData& data) {
  if (data.sequence_index < 0 || data.sequence_index > 0b0011'1111'1111'1111) {
    return std::nullopt;
  }
  // Frame instrumentation data must have sample values.
  if (data.sample_values.empty()) {
    return std::nullopt;
  }
  return CorruptionDetectionMessage::Builder()
      .WithSequenceIndex(GetSequenceIndexForMessage(
          data.sequence_index, data.communicate_upper_bits))
      .WithInterpretSequenceIndexAsMostSignificantBits(
          data.communicate_upper_bits)
      .WithStdDev(data.std_dev)
      .WithLumaErrorThreshold(data.luma_error_threshold)
      .WithChromaErrorThreshold(data.chroma_error_threshold)
      .WithSampleValues(data.sample_values)
      .Build();
}

std::optional<CorruptionDetectionMessage>
ConvertFrameInstrumentationSyncDataToCorruptionDetectionMessage(
    const FrameInstrumentationSyncData& data) {
  RTC_DCHECK(data.communicate_upper_bits)
      << "FrameInstrumentationSyncData data must always send the upper bits.";

  if (data.sequence_index < 0 || data.sequence_index > 0b0011'1111'1111'1111) {
    return std::nullopt;
  }
  return CorruptionDetectionMessage::Builder()
      .WithSequenceIndex(GetSequenceIndexForMessage(
          data.sequence_index, data.communicate_upper_bits))
      .WithInterpretSequenceIndexAsMostSignificantBits(true)
      .Build();
}

}  // namespace webrtc
