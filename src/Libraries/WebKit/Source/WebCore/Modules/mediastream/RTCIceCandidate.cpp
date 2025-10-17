/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#include "RTCIceCandidate.h"

#if ENABLE(WEB_RTC)

#include "RTCIceCandidateInit.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RTCIceCandidate);

RTCIceCandidate::RTCIceCandidate(const String& candidate, const String& sdpMid, std::optional<unsigned short> sdpMLineIndex, Fields&& fields)
    : m_candidate(candidate)
    , m_sdpMid(sdpMid)
    , m_sdpMLineIndex(sdpMLineIndex)
    , m_fields(WTFMove(fields))
{
    ASSERT(!sdpMid.isNull() || sdpMLineIndex);
}

ExceptionOr<Ref<RTCIceCandidate>> RTCIceCandidate::create(const RTCIceCandidateInit& dictionary)
{
    if (dictionary.sdpMid.isNull() && !dictionary.sdpMLineIndex)
        return Exception { ExceptionCode::TypeError, "Candidate must not have both null sdpMid and sdpMLineIndex"_s };

    auto fields = valueOrDefault(parseIceCandidateSDP(dictionary.candidate));
    fields.usernameFragment = dictionary.usernameFragment;
    return adoptRef(*new RTCIceCandidate(dictionary.candidate, dictionary.sdpMid, dictionary.sdpMLineIndex, WTFMove(fields)));
}

Ref<RTCIceCandidate> RTCIceCandidate::create(const String& candidate, const String& sdpMid, std::optional<unsigned short> sdpMLineIndex)
{
    auto fields = parseIceCandidateSDP(candidate);
    return adoptRef(*new RTCIceCandidate(candidate, sdpMid, sdpMLineIndex, WTFMove(*fields)));
}

RTCIceCandidateInit RTCIceCandidate::toJSON() const
{
    RTCIceCandidateInit result;
    result.candidate = m_candidate;
    result.sdpMid = m_sdpMid;
    result.sdpMLineIndex = m_sdpMLineIndex;
    return result;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
