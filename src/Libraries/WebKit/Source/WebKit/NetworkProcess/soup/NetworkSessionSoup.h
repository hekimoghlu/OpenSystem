/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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

#include "NetworkSession.h"
#include "SoupCookiePersistentStorageType.h"
#include "WebPageProxyIdentifier.h"
#include <wtf/TZoneMalloc.h>

typedef struct _SoupSession SoupSession;

namespace WebCore {
class CertificateInfo;
class SoupNetworkSession;
struct SoupNetworkProxySettings;
}

namespace WebKit {

class NetworkSocketChannel;
class WebSocketTask;
struct NetworkSessionCreationParameters;

class NetworkSessionSoup final : public NetworkSession {
    WTF_MAKE_TZONE_ALLOCATED(NetworkSessionSoup);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(NetworkSessionSoup);
public:
    static std::unique_ptr<NetworkSession> create(NetworkProcess& networkProcess, const NetworkSessionCreationParameters& parameters)
    {
        return makeUnique<NetworkSessionSoup>(networkProcess, parameters);
    }
    NetworkSessionSoup(NetworkProcess&, const NetworkSessionCreationParameters&);
    ~NetworkSessionSoup();

    WebCore::SoupNetworkSession& soupNetworkSession() const { return *m_networkSession; }
    SoupSession* soupSession() const;

    void setCookiePersistentStorage(const String& storagePath, SoupCookiePersistentStorageType);

    void setPersistentCredentialStorageEnabled(bool enabled) { m_persistentCredentialStorageEnabled = enabled; }
    bool persistentCredentialStorageEnabled() const { return m_persistentCredentialStorageEnabled; }

    void setIgnoreTLSErrors(bool);
    void allowSpecificHTTPSCertificateForHost(const WebCore::CertificateInfo&, const String&);
    void setProxySettings(const WebCore::SoupNetworkProxySettings&);

private:
    std::unique_ptr<WebSocketTask> createWebSocketTask(WebPageProxyIdentifier, std::optional<WebCore::FrameIdentifier>, std::optional<WebCore::PageIdentifier>, NetworkSocketChannel&, const WebCore::ResourceRequest&, const String& protocol, const WebCore::ClientOrigin&, bool, bool, OptionSet<WebCore::AdvancedPrivacyProtections>, WebCore::StoredCredentialsPolicy) final;
    void clearCredentials(WallTime) final;

    std::unique_ptr<WebCore::SoupNetworkSession> m_networkSession;
    bool m_persistentCredentialStorageEnabled { true };
};

} // namespace WebKit
