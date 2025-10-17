/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 7, 2024.
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

#include "WebRTCMonitor.h"
#include <WebCore/LibWebRTCProvider.h>
#include <WebCore/ProcessQualified.h>
#include <WebCore/RTCNetworkManager.h>
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class LibWebRTCNetworkManager final : public WebCore::RTCNetworkManager, public rtc::NetworkManagerBase, public webrtc::MdnsResponderInterface, public WebRTCMonitorObserver {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCNetworkManager);
public:
    static RefPtr<LibWebRTCNetworkManager> getOrCreate(WebCore::ScriptExecutionContextIdentifier);
    ~LibWebRTCNetworkManager();

    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    void setEnumeratingAllNetworkInterfacesEnabled(bool);
    void setEnumeratingVisibleNetworkInterfacesEnabled(bool);

    static void signalUsedInterface(WebCore::ScriptExecutionContextIdentifier, String&&);

private:
    explicit LibWebRTCNetworkManager(WebCore::ScriptExecutionContextIdentifier);

    // WebCore::RTCNetworkManager
    void setICECandidateFiltering(bool doFiltering) final { m_useMDNSCandidates = doFiltering; }
    void unregisterMDNSNames() final;
    void close() final;
    const String& interfaceNameForTesting() const final;

    // webrtc::NetworkManagerBase
    void StartUpdating() final;
    void StopUpdating() final;
    webrtc::MdnsResponderInterface* GetMdnsResponder() const final;

    // webrtc::MdnsResponderInterface
    void CreateNameForAddress(const rtc::IPAddress&, NameCreatedCallback);
    void RemoveNameForAddress(const rtc::IPAddress&, NameRemovedCallback);

    // WebRTCMonitorObserver
    void networksChanged(const Vector<RTCNetwork>&, const RTCNetwork::IPAddress&, const RTCNetwork::IPAddress&) final;
    void networkProcessCrashed() final;

    void networksChanged(const Vector<RTCNetwork>&, const RTCNetwork::IPAddress&, const RTCNetwork::IPAddress&, bool forceSignaling);
    void signalUsedInterface(String&&);

    WebCore::ScriptExecutionContextIdentifier m_documentIdentifier;
    bool m_useMDNSCandidates { true };
    bool m_receivedNetworkList { false };
#if ASSERT_ENABLED
    bool m_isClosed { false };
#endif
    bool m_enableEnumeratingAllNetworkInterfaces { false };
    bool m_enableEnumeratingVisibleNetworkInterfaces { false };
    bool m_hasQueriedInterface { false };
    HashSet<String> m_allowedInterfaces;
};

} // namespace WebKit

#endif
