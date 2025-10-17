/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 1, 2023.
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
#ifndef MEDIA_BASE_SDP_VIDEO_FORMAT_UTILS_H_
#define MEDIA_BASE_SDP_VIDEO_FORMAT_UTILS_H_

#include <optional>

#include "api/video_codecs/sdp_video_format.h"

namespace webrtc {
// Generate codec parameters that will be used as answer in an SDP negotiation
// based on local supported parameters and remote offered parameters. Both
// `local_supported_params`, `remote_offered_params`, and `answer_params`
// represent sendrecv media descriptions, i.e they are a mix of both encode and
// decode capabilities. In theory, when the profile in `local_supported_params`
// represent a strict superset of the profile in `remote_offered_params`, we
// could limit the profile in `answer_params` to the profile in
// `remote_offered_params`. However, to simplify the code, each supported H264
// profile should be listed explicitly in the list of local supported codecs,
// even if they are redundant. Then each local codec in the list should be
// tested one at a time against the remote codec, and only when the profiles are
// equal should this function be called. Therefore, this function does not need
// to handle profile intersection, and the profile of `local_supported_params`
// and `remote_offered_params` must be equal before calling this function. The
// parameters that are used when negotiating are the level part of
// profile-level-id and level-asymmetry-allowed.
void H264GenerateProfileLevelIdForAnswer(
    const CodecParameterMap& local_supported_params,
    const CodecParameterMap& remote_offered_params,
    CodecParameterMap* answer_params);

#ifdef RTC_ENABLE_H265
// Works similarly as H264GenerateProfileLevelIdForAnswer, but generates codec
// parameters that will be used as answer for H.265.
// Media configuration parameters, except level-id, must be used symmetrically.
// For level-id, the highest level indicated by the answer must not be higher
// than that indicated by the offer.
void H265GenerateProfileTierLevelForAnswer(
    const CodecParameterMap& local_supported_params,
    const CodecParameterMap& remote_offered_params,
    CodecParameterMap* answer_params);
#endif

// Parse max frame rate from SDP FMTP line. std::nullopt is returned if the
// field is missing or not a number.
std::optional<int> ParseSdpForVPxMaxFrameRate(const CodecParameterMap& params);

// Parse max frame size from SDP FMTP line. std::nullopt is returned if the
// field is missing or not a number. Please note that the value is stored in sub
// blocks but the returned value is in total number of pixels.
std::optional<int> ParseSdpForVPxMaxFrameSize(const CodecParameterMap& params);

// Determines whether the non-standard x-google-per-layer-pli fmtp is present
// in the parameters and has a value of "1".
bool SupportsPerLayerPictureLossIndication(const CodecParameterMap& params);

}  // namespace webrtc

#endif  // MEDIA_BASE_SDP_VIDEO_FORMAT_UTILS_H_
