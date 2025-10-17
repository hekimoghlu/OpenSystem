/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 22, 2022.
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
#include "WebRTCMonitor.h"

#if USE(LIBWEBRTC)

#include "Logging.h"
#include "NetworkConnectionToWebProcessMessages.h"
#include "NetworkProcessConnection.h"
#include "NetworkRTCMonitorMessages.h"
#include "WebProcess.h"
#include <wtf/MainThread.h>

namespace WebKit {

#define WEBRTC_RELEASE_LOG(fmt, ...) RELEASE_LOG(Network, "%p - WebRTCMonitor::" fmt, this, ##__VA_ARGS__)

WebRTCMonitor::WebRTCMonitor(LibWebRTCNetwork& libWebRTCNetwork)
    : m_libWebRTCNetwork(libWebRTCNetwork)
{
}

void WebRTCMonitor::ref() const
{
    m_libWebRTCNetwork->ref();
}

void WebRTCMonitor::deref() const
{
    m_libWebRTCNetwork->deref();
}

void WebRTCMonitor::startUpdating()
{
    WEBRTC_RELEASE_LOG("StartUpdating - Asking network process to start updating");

    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkRTCMonitor::StartUpdatingIfNeeded(), 0);
    ++m_clientCount;
}

void WebRTCMonitor::stopUpdating()
{
    WEBRTC_RELEASE_LOG("StopUpdating");

    ASSERT(m_clientCount);
    if (--m_clientCount)
        return;

    WEBRTC_RELEASE_LOG("StopUpdating - Asking network process to stop updating");
    WebProcess::singleton().ensureNetworkProcessConnection().connection().send(Messages::NetworkRTCMonitor::StopUpdating(), 0);
}

void WebRTCMonitor::networksChanged(Vector<RTCNetwork>&& networkList, RTCNetwork::IPAddress&& ipv4, RTCNetwork::IPAddress&& ipv6)
{
    ASSERT(isMainRunLoop());
    WEBRTC_RELEASE_LOG("NetworksChanged");

    m_didReceiveNetworkList = true;
    m_networkList = WTFMove(networkList);
    m_ipv4 = WTFMove(ipv4);
    m_ipv6 = WTFMove(ipv6);

    for (Ref observer : m_observers)
        observer->networksChanged(m_networkList, m_ipv4, m_ipv6);
}

void WebRTCMonitor::networkProcessCrashed()
{
    for (Ref observer : m_observers)
        observer->networkProcessCrashed();
}

} // namespace WebKit

#undef WEBRTC_RELEASE_LOG

#endif // USE(LIBWEBRTC)
