/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
#include <wtf/CheckedRef.h>
#include <wtf/WeakPtr.h>

namespace WebKit {
class NetworkRTCMonitor;
}

namespace IPC {
class Connection;
class Decoder;
}

namespace WebKit {

class NetworkRTCProvider;

class NetworkRTCMonitor final : public CanMakeWeakPtr<NetworkRTCMonitor> {
public:
    explicit NetworkRTCMonitor(NetworkRTCProvider&);
    ~NetworkRTCMonitor();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);
    void stopUpdating();
    bool isStarted() const { return m_isStarted; }
    NetworkRTCProvider& rtcProvider();

    void onNetworksChanged(const Vector<RTCNetwork>&, const RTCNetwork::IPAddress&, const RTCNetwork::IPAddress&);

    const RTCNetwork::IPAddress& ipv4() const;
    const RTCNetwork::IPAddress& ipv6()  const;

    void ref();
    void deref();

private:
    void startUpdatingIfNeeded();

    CheckedRef<NetworkRTCProvider> m_rtcProvider;
    bool m_isStarted { false };
};

} // namespace WebKit

#endif // USE(LIBWEBRTC)
