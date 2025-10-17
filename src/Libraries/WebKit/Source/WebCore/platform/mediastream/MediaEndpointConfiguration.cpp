/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#include "MediaEndpointConfiguration.h"

#if ENABLE(WEB_RTC)

namespace WebCore {

MediaEndpointConfiguration::MediaEndpointConfiguration(Vector<IceServerInfo>&& iceServers, RTCIceTransportPolicy iceTransportPolicy, RTCBundlePolicy bundlePolicy, RTCPMuxPolicy rtcpMuxPolicy, unsigned short iceCandidatePoolSize, Vector<CertificatePEM>&& certificates)
    : iceServers(WTFMove(iceServers))
    , iceTransportPolicy(iceTransportPolicy)
    , bundlePolicy(bundlePolicy)
    , rtcpMuxPolicy(rtcpMuxPolicy)
    , iceCandidatePoolSize(iceCandidatePoolSize)
    , certificates(WTFMove(certificates))
{
}

MediaEndpointConfiguration::IceServerInfo::IceServerInfo(Vector<URL>&& urls, const String& credential, const String& username)
    : urls(WTFMove(urls))
    , credential(credential)
    , username(username)
{
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
