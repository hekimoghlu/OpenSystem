/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#ifndef MEDIA_BASE_CODEC_COMPARATORS_H_
#define MEDIA_BASE_CODEC_COMPARATORS_H_

#include <optional>
#include <vector>

#include "api/rtp_parameters.h"
#include "media/base/codec.h"

namespace webrtc {

// Comparison used in the PayloadTypePicker
bool MatchesForSdp(const cricket::Codec& codec_1,
                   const cricket::Codec& codec_2);

// Comparison used for the Codec::Matches function
bool MatchesWithCodecRules(const cricket::Codec& left_codec,
                           const cricket::Codec& codec);

// Finds a codec in `codecs2` that matches `codec_to_match`, which is
// a member of `codecs1`. If `codec_to_match` is an RED or RTX codec, both
// the codecs themselves and their associated codecs must match.
// The purpose of this function is that codecs1 and codecs2 are different
// PT numbering spaces, and it is trying to find the codec in codecs2
// that has the same functionality as `codec_to_match` so that its PT
// can be used in place of the original.
std::optional<cricket::Codec> FindMatchingCodec(
    const std::vector<cricket::Codec>& codecs1,
    const std::vector<cricket::Codec>& codecs2,
    const cricket::Codec& codec_to_match);

// Similar to `Codec::MatchesRtpCodec` but not an exact match of parameters.
// Unspecified parameters are treated as default.
bool IsSameRtpCodec(const cricket::Codec& codec, const RtpCodec& rtp_codec);

}  // namespace webrtc

#endif  // MEDIA_BASE_CODEC_COMPARATORS_H_
