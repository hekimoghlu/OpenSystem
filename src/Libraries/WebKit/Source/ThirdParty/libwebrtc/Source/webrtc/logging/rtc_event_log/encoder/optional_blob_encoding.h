/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#ifndef LOGGING_RTC_EVENT_LOG_ENCODER_OPTIONAL_BLOB_ENCODING_H_
#define LOGGING_RTC_EVENT_LOG_ENCODER_OPTIONAL_BLOB_ENCODING_H_

#include <stddef.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace webrtc {

// Encode a sequence of optional strings, whose length is not known to be
// discernable from the blob itself (i.e. without being transmitted OOB),
// in a way that would allow us to separate them again on the decoding side.
// EncodeOptionalBlobs() may not fail but may return an empty string
std::string EncodeOptionalBlobs(
    const std::vector<std::optional<std::string>>& blobs);

// Calling DecodeOptionalBlobs() on an empty string, or with `num_of_blobs` set
// to 0, is an error. DecodeOptionalBlobs() returns an empty vector if it fails,
// which can happen if `encoded_blobs` is corrupted.
std::vector<std::optional<std::string>> DecodeOptionalBlobs(
    absl::string_view encoded_blobs,
    size_t num_of_blobs);

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_ENCODER_OPTIONAL_BLOB_ENCODING_H_
