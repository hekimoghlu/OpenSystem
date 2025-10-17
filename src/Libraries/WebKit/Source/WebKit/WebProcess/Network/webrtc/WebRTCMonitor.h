/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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

#if USE(LIBWEBRTC)

#include "RTCNetwork.h"
#include <WebCore/LibWebRTCProvider.h>
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/Forward.h>
#include <wtf/WeakHashSet.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class LibWebRTCNetwork;
struct NetworksChangedData;

class WebRTCMonitorObserver : public AbstractRefCountedAndCanMakeWeakPtr<WebRTCMonitorObserver> {
public:
    virtual ~WebRTCMonitorObserver() = default;

    virtual void networksChanged(const Vector<RTCNetwork>&, const RTCNetwork::IPAddress&, const RTCNetwork::IPAddress&) = 0;
    virtual void networkProcessCrashed() = 0;
};

class WebRTCMonitor {
public:
    explicit WebRTCMonitor(LibWebRTCNetwork&);

    void ref() const;
    void deref() const;

    void addObserver(WebRTCMonitorObserver& observer) { m_observers.add(observer); }
    void removeObserver(WebRTCMonitorObserver& observer) { m_observers.remove(observer); }
    void startUpdating();
    void stopUpdating();

    void networkProcessCrashed();
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    bool didReceiveNetworkList() const { return m_didReceiveNetworkList; }
    const Vector<RTCNetwork>& networkList() const { return m_networkList; }
    const RTCNetwork::IPAddress& ipv4() const { return m_ipv4; }
    const RTCNetwork::IPAddress& ipv6() const { return m_ipv6; }

private:
    void networksChanged(Vector<RTCNetwork>&&, RTCNetwork::IPAddress&&, RTCNetwork::IPAddress&&);

    WeakRef<LibWebRTCNetwork> m_libWebRTCNetwork;
    unsigned m_clientCount { 0 };
    WeakHashSet<WebRTCMonitorObserver> m_observers;
    bool m_didReceiveNetworkList { false };
    Vector<RTCNetwork> m_networkList;
    RTCNetwork::IPAddress m_ipv4;
    RTCNetwork::IPAddress m_ipv6;
};

} // namespace WebKit

#endif // USE(LIBWEBRTC)
