/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

#include "RTCBundlePolicy.h"
#include "RTCIceTransportPolicy.h"
#include "RTCPMuxPolicy.h"
#include <wtf/URL.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

struct MediaEndpointConfiguration {
    // FIXME: We might be able to remove these constructors once all compilers can handle without it (see https://bugs.webkit.org/show_bug.cgi?id=163255#c15)
    struct IceServerInfo {
        Vector<URL> urls;
        String credential;
        String username;

        IceServerInfo(Vector<URL>&&, const String&, const String&);
    };
    struct CertificatePEM {
        String certificate;
        String privateKey;
    };

    MediaEndpointConfiguration(Vector<IceServerInfo>&&, RTCIceTransportPolicy, RTCBundlePolicy, RTCPMuxPolicy, unsigned short, Vector<CertificatePEM>&&);

    Vector<IceServerInfo> iceServers;
    RTCIceTransportPolicy iceTransportPolicy;
    RTCBundlePolicy bundlePolicy;
    RTCPMuxPolicy rtcpMuxPolicy;
    unsigned short iceCandidatePoolSize;
    Vector<CertificatePEM> certificates;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
