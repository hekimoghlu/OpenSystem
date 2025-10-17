/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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

#if ENABLE(WEB_AUTHN)

#include "APIObject.h"
#include <WebCore/AuthenticatorTransport.h>
#include <variant>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
enum class ClientDataType : bool;

class AuthenticatorResponse;

struct ExceptionData;
struct MockWebAuthenticationConfiguration;
}

namespace WebKit {
class AuthenticatorManager;

struct WebAuthenticationRequestData;
}

namespace API {

class WebAuthenticationPanelClient;

class WebAuthenticationPanel final : public ObjectImpl<Object::Type::WebAuthenticationPanel>, public CanMakeWeakPtr<WebAuthenticationPanel> {
public:
    using Response = std::variant<Ref<WebCore::AuthenticatorResponse>, WebCore::ExceptionData>;
    using Callback = CompletionHandler<void(Response&&)>;

    WebAuthenticationPanel();
    ~WebAuthenticationPanel();

    void handleRequest(WebKit::WebAuthenticationRequestData&&, Callback&&);
    void cancel() const;
    void setMockConfiguration(WebCore::MockWebAuthenticationConfiguration&&);

    const WebAuthenticationPanelClient& client() const { return m_client.get(); }
    void setClient(Ref<WebAuthenticationPanelClient>&&);

    // FIXME: <rdar://problem/71509848> Remove the following deprecated methods.
    using TransportSet = HashSet<WebCore::AuthenticatorTransport, WTF::IntHash<WebCore::AuthenticatorTransport>, WTF::StrongEnumHashTraits<WebCore::AuthenticatorTransport>>;
    static Ref<WebAuthenticationPanel> create(const WebKit::AuthenticatorManager&, const WTF::String& rpId, const TransportSet&, WebCore::ClientDataType, const WTF::String& userName);
    WTF::String rpId() const { return m_rpId; }
    const Vector<WebCore::AuthenticatorTransport>& transports() const { return m_transports; }
    WebCore::ClientDataType clientDataType() const { return m_clientDataType; }
    WTF::String userName() const { return m_userName; }

private:
    // FIXME: <rdar://problem/71509848> Remove the following deprecated method.
    WebAuthenticationPanel(const WebKit::AuthenticatorManager&, const WTF::String& rpId, const TransportSet&, WebCore::ClientDataType, const WTF::String& userName);

    RefPtr<WebKit::AuthenticatorManager> protectedManager() const;

    RefPtr<WebKit::AuthenticatorManager> m_manager; // FIXME: <rdar://problem/71509848> Change to Ref.
    Ref<WebAuthenticationPanelClient> m_client;

    // FIXME: <rdar://problem/71509848> Remove the following deprecated fields.
    WeakPtr<WebKit::AuthenticatorManager> m_weakManager;
    WTF::String m_rpId;
    Vector<WebCore::AuthenticatorTransport> m_transports;
    WebCore::ClientDataType m_clientDataType;
    WTF::String m_userName;
};

} // namespace API

#endif // ENABLE(WEB_AUTHN)
