/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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

#include "IceCandidate.h"
#include "MediaPayload.h"
#include "RealtimeMediaSource.h"
#include <wtf/Vector.h>

namespace WebCore {

struct PeerMediaDescription {
    void addPayload(MediaPayload&& payload) { payloads.append(WTFMove(payload)); }
    void addSsrc(unsigned ssrc) { ssrcs.append(ssrc); }
    void clearSsrcs() { ssrcs.clear(); }
    void addIceCandidate(IceCandidate&& candidate) { iceCandidates.append(WTFMove(candidate)); }

    String type;
    unsigned short port { 9 };
    String address { "0.0.0.0"_s };
    String mode { "sendrecv"_s };
    String mid;

    Vector<MediaPayload> payloads;

    bool rtcpMux { true };
    String rtcpAddress;
    unsigned short rtcpPort { 0 };

    String mediaStreamId;
    String mediaStreamTrackId;

    String dtlsSetup { "actpass"_s };
    String dtlsFingerprintHashFunction;
    String dtlsFingerprint;

    Vector<unsigned> ssrcs;
    String cname;

    String iceUfrag;
    String icePassword;
    Vector<IceCandidate> iceCandidates;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
