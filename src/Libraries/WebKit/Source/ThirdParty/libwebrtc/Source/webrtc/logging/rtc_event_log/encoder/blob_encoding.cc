/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
#include "logging/rtc_event_log/encoder/blob_encoding.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/string_view.h"
#include "logging/rtc_event_log/encoder/var_int.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {

std::string EncodeBlobs(const std::vector<std::string>& blobs) {
  RTC_DCHECK(!blobs.empty());

  size_t result_length_bound = kMaxVarIntLengthBytes * blobs.size();
  for (const auto& blob : blobs) {
    // Providing an input so long that it would cause a wrap-around is an error.
    RTC_DCHECK_GE(result_length_bound + blob.length(), result_length_bound);
    result_length_bound += blob.length();
  }

  std::string result;
  result.reserve(result_length_bound);

  // First, encode all of the lengths.
  for (absl::string_view blob : blobs) {
    result += EncodeVarInt(blob.length());
  }

  // Second, encode the actual blobs.
  for (absl::string_view blob : blobs) {
    result.append(blob.data(), blob.length());
  }

  RTC_DCHECK_LE(result.size(), result_length_bound);
  return result;
}

std::vector<absl::string_view> DecodeBlobs(absl::string_view encoded_blobs,
                                           size_t num_of_blobs) {
  if (encoded_blobs.empty()) {
    RTC_LOG(LS_WARNING) << "Corrupt input; empty input.";
    return std::vector<absl::string_view>();
  }

  if (num_of_blobs == 0u) {
    RTC_LOG(LS_WARNING)
        << "Corrupt input; number of blobs must be greater than 0.";
    return std::vector<absl::string_view>();
  }

  // Read the lengths of all blobs.
  std::vector<uint64_t> lengths(num_of_blobs);
  for (size_t i = 0; i < num_of_blobs; ++i) {
    bool success = false;
    std::tie(success, encoded_blobs) = DecodeVarInt(encoded_blobs, &lengths[i]);
    if (!success) {
      RTC_LOG(LS_WARNING) << "Corrupt input; varint decoding failed.";
      return std::vector<absl::string_view>();
    }
  }

  // Read the blobs themselves.
  std::vector<absl::string_view> blobs(num_of_blobs);
  for (size_t i = 0; i < num_of_blobs; ++i) {
    if (lengths[i] > encoded_blobs.length()) {
      RTC_LOG(LS_WARNING) << "Corrupt input; blob sizes exceed input size.";
      return std::vector<absl::string_view>();
    }

    blobs[i] = encoded_blobs.substr(0, lengths[i]);
    encoded_blobs = encoded_blobs.substr(lengths[i]);
  }

  if (!encoded_blobs.empty()) {
    RTC_LOG(LS_WARNING) << "Corrupt input; unrecognized trailer.";
    return std::vector<absl::string_view>();
  }

  return blobs;
}

}  // namespace webrtc
