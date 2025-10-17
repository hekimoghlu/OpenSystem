/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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

#include "IdentifierTypes.h"
#include "MessageReceiver.h"
#include "NetworkProcessSupplement.h"
#include "WebPageProxyIdentifier.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class MessageSender;
}

namespace PAL {
class SessionID;
}

namespace WebCore {
class AuthenticationChallenge;
class Credential;
class SecurityOriginData;
}

namespace WebKit {

class Download;
class NetworkProcess;
class WebFrame;
enum class NegotiatedLegacyTLS : bool { No, Yes };

enum class AuthenticationChallengeDisposition : uint8_t;
using ChallengeCompletionHandler = CompletionHandler<void(AuthenticationChallengeDisposition, const WebCore::Credential&)>;

class AuthenticationManager : public NetworkProcessSupplement, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(AuthenticationManager);
    WTF_MAKE_NONCOPYABLE(AuthenticationManager);
public:
    explicit AuthenticationManager(NetworkProcess&);
    ~AuthenticationManager();

    void ref() const final;
    void deref() const final;

    static ASCIILiteral supplementName();

    void didReceiveAuthenticationChallenge(PAL::SessionID, std::optional<WebPageProxyIdentifier>, const WebCore::SecurityOriginData*, const WebCore::AuthenticationChallenge&, NegotiatedLegacyTLS, ChallengeCompletionHandler&&);
    void didReceiveAuthenticationChallenge(IPC::MessageSender& download, const WebCore::AuthenticationChallenge&, ChallengeCompletionHandler&&);

    void completeAuthenticationChallenge(AuthenticationChallengeIdentifier, AuthenticationChallengeDisposition, WebCore::Credential&&);

    void negotiatedLegacyTLS(WebPageProxyIdentifier) const;

private:
    Ref<NetworkProcess> protectedProcess() const;
    struct Challenge;

#if HAVE(SEC_KEY_PROXY)
    // NetworkProcessSupplement
    void initializeConnection(IPC::Connection*) final;
#endif

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    AuthenticationChallengeIdentifier addChallengeToChallengeMap(UniqueRef<Challenge>&&);
    bool shouldCoalesceChallenge(std::optional<WebPageProxyIdentifier>, AuthenticationChallengeIdentifier, const WebCore::AuthenticationChallenge&) const;

    Vector<AuthenticationChallengeIdentifier> coalesceChallengesMatching(AuthenticationChallengeIdentifier) const;

    WeakRef<NetworkProcess> m_process;

    HashMap<AuthenticationChallengeIdentifier, UniqueRef<Challenge>> m_challenges;
};

} // namespace WebKit
