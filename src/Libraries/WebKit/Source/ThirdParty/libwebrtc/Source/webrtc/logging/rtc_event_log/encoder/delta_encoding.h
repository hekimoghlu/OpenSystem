/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
#ifndef LOGGING_RTC_EVENT_LOG_ENCODER_DELTA_ENCODING_H_
#define LOGGING_RTC_EVENT_LOG_ENCODER_DELTA_ENCODING_H_

#include <stddef.h>
#include <stdint.h>

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace webrtc {

// Encode `values` as a sequence of deltas following on `base` and return it.
// If all of the values were equal to the base, an empty string will be
// returned; this is a valid encoding of that edge case.
// `base` is not guaranteed to be written into `output`, and must therefore
// be provided separately to the decoder.
// This function never fails.
// TODO(eladalon): Split into optional and non-optional variants (efficiency).
std::string EncodeDeltas(std::optional<uint64_t> base,
                         const std::vector<std::optional<uint64_t>>& values);

// EncodeDeltas() and DecodeDeltas() are inverse operations;
// invoking DecodeDeltas() over the output of EncodeDeltas(), will return
// the input originally given to EncodeDeltas().
// `num_of_deltas` must be greater than zero. If input is not a valid encoding
// of `num_of_deltas` elements based on `base`, the function returns an empty
// vector, which signals an error.
// TODO(eladalon): Split into optional and non-optional variants (efficiency).
std::vector<std::optional<uint64_t>> DecodeDeltas(absl::string_view input,
                                                  std::optional<uint64_t> base,
                                                  size_t num_of_deltas);

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_ENCODER_DELTA_ENCODING_H_
