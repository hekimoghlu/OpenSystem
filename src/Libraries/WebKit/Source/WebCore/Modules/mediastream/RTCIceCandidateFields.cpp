/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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
#include "config.h"
#include "RTCIceCandidateFields.h"

#if USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "LibWebRTCUtils.h"

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/pc/webrtc_sdp.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

std::optional<RTCIceCandidateFields> parseIceCandidateSDP(const String& sdp)
{
    cricket::Candidate candidate;
    if (!webrtc::ParseCandidate(sdp.utf8().data(), &candidate, nullptr, true))
        return { };

    return convertIceCandidate(candidate);
}

} // namespace WebCore

#endif // USE(LIBWEBRTC)
