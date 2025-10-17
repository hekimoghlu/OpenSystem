/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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
#pragma once

#if ENABLE(WEB_RTC)

#include "ExceptionOr.h"
#include "RTCIceCandidateFields.h"
#include "ScriptWrappable.h"

namespace WebCore {

struct RTCIceCandidateInit;

class RTCIceCandidate final : public RefCounted<RTCIceCandidate>, public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RTCIceCandidate);
public:
    using Fields = RTCIceCandidateFields;

    static ExceptionOr<Ref<RTCIceCandidate>> create(const RTCIceCandidateInit&);
    static Ref<RTCIceCandidate> create(const String& candidate, const String& sdpMid, std::optional<unsigned short> sdpMLineIndex);
    static Ref<RTCIceCandidate> create(const String& candidate, const String& sdpMid, Fields&& fields) { return adoptRef(*new RTCIceCandidate(candidate, sdpMid, { }, WTFMove(fields))); }

    const String& candidate() const { return m_candidate; }
    const String& sdpMid() const { return m_sdpMid; }
    std::optional<unsigned short> sdpMLineIndex() const { return m_sdpMLineIndex; }

    String foundation() const { return m_fields.foundation; }
    std::optional<RTCIceComponent> component() const { return m_fields.component; }
    std::optional<unsigned> priority() const { return m_fields.priority; }
    String address() const { return m_fields.address; }
    std::optional<RTCIceProtocol> protocol() const { return m_fields.protocol; }
    std::optional<unsigned short> port() const { return m_fields.port; }
    std::optional<RTCIceCandidateType> type() const { return m_fields.type; }
    std::optional<RTCIceTcpCandidateType> tcpType() const { return m_fields.tcpType; }
    String relatedAddress() const { return m_fields.relatedAddress; }
    std::optional<unsigned short> relatedPort() const { return m_fields.relatedPort; }

    String usernameFragment() const { return m_fields.usernameFragment; }
    RTCIceCandidateInit toJSON() const;

private:
    RTCIceCandidate(const String& candidate, const String& sdpMid, std::optional<unsigned short> sdpMLineIndex, Fields&&);

    String m_candidate;
    String m_sdpMid;
    std::optional<unsigned short> m_sdpMLineIndex;
    Fields m_fields;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
