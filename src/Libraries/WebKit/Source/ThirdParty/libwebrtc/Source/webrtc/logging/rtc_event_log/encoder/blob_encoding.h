/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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
#ifndef LOGGING_RTC_EVENT_LOG_ENCODER_BLOB_ENCODING_H_
#define LOGGING_RTC_EVENT_LOG_ENCODER_BLOB_ENCODING_H_

#include <stddef.h>

#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace webrtc {

// Encode/decode a sequence of strings, whose length is not known to be
// discernable from the blob itself (i.e. without being transmitted OOB),
// in a way that would allow us to separate them again on the decoding side.
// The number of blobs is assumed to be transmitted OOB. For example, if
// multiple sequences of different blobs are sent, but all sequences contain
// the same number of blobs, it is beneficial to not encode the number of blobs.
//
// EncodeBlobs() must be given a non-empty vector. The blobs themselves may
// be equal to "", though.
// EncodeBlobs() may not fail.
// EncodeBlobs() never returns the empty string.
//
// Calling DecodeBlobs() on an empty string, or with `num_of_blobs` set to 0,
// is an error.
// DecodeBlobs() returns an empty vector if it fails, e.g. due to a mismatch
// between `num_of_blobs` and `encoded_blobs`, which can happen if
// `encoded_blobs` is corrupted.
// When successful, DecodeBlobs() returns a vector of string_view objects,
// which refer to the original input (`encoded_blobs`), and therefore may
// not outlive it.
//
// Note that the returned std::string might have been reserved for significantly
// more memory than it ends up using. If the caller to EncodeBlobs() intends
// to store the result long-term, they should consider shrink_to_fit()-ing it.
std::string EncodeBlobs(const std::vector<std::string>& blobs);
std::vector<absl::string_view> DecodeBlobs(absl::string_view encoded_blobs,
                                           size_t num_of_blobs);

}  // namespace webrtc

#endif  // LOGGING_RTC_EVENT_LOG_ENCODER_BLOB_ENCODING_H_
