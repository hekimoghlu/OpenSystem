/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
#include "AuthenticationManager.h"

#include "AuthenticationChallengeDisposition.h"
#include "AuthenticationManagerMessages.h"
#include "Download.h"
#include "DownloadProxyMessages.h"
#include "MessageSenderInlines.h"
#include "NetworkProcess.h"
#include "NetworkProcessProxyMessages.h"
#include "PendingDownload.h"
#include "WebFrame.h"
#include "WebPage.h"
#include "WebPageProxyMessages.h"
#include <WebCore/AuthenticationChallenge.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

struct AuthenticationManager::Challenge {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    Challenge(std::optional<WebPageProxyIdentifier> pageID, const WebCore::AuthenticationChallenge& challenge, ChallengeCompletionHandler&& completionHandler)
        : pageID(pageID)
        , challenge(challenge)
        , completionHandler(WTFMove(completionHandler)) { }

    Markable<WebPageProxyIdentifier> pageID;
    WebCore::AuthenticationChallenge challenge;
    ChallengeCompletionHandler completionHandler;
};

static bool canCoalesceChallenge(const WebCore::AuthenticationChallenge& challenge)
{
    // Do not coalesce server trust evaluation requests because ProtectionSpace comparison does not evaluate server trust (e.g. certificate).
    return challenge.protectionSpace().authenticationScheme() != ProtectionSpace::AuthenticationScheme::ServerTrustEvaluationRequested;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(AuthenticationManager);

ASCIILiteral AuthenticationManager::supplementName()
{
    return "AuthenticationManager"_s;
}

AuthenticationManager::AuthenticationManager(NetworkProcess& process)
    : m_process(process)
{
    process.addMessageReceiver(Messages::AuthenticationManager::messageReceiverName(), *this);
}

AuthenticationManager::~AuthenticationManager() = default;

void AuthenticationManager::ref() const
{
    return m_process->ref();
}

void AuthenticationManager::deref() const
{
    return m_process->deref();
}

inline Ref<NetworkProcess> AuthenticationManager::protectedProcess() const
{
    ASSERT(RunLoop::isMain());
    return m_process.get();
}

AuthenticationChallengeIdentifier AuthenticationManager::addChallengeToChallengeMap(UniqueRef<Challenge>&& challenge)
{
    ASSERT(RunLoop::isMain());

    auto challengeID = AuthenticationChallengeIdentifier::generate();
    m_challenges.set(challengeID, WTFMove(challenge));
    return challengeID;
}

bool AuthenticationManager::shouldCoalesceChallenge(std::optional<WebPageProxyIdentifier> pageID, AuthenticationChallengeIdentifier challengeID, const AuthenticationChallenge& challenge) const
{
    if (!canCoalesceChallenge(challenge))
        return false;

    for (auto& item : m_challenges) {
        if (item.key != challengeID && item.value->pageID == pageID && ProtectionSpace::compare(challenge.protectionSpace(), item.value->challenge.protectionSpace()))
            return true;
    }
    return false;
}

Vector<AuthenticationChallengeIdentifier> AuthenticationManager::coalesceChallengesMatching(AuthenticationChallengeIdentifier challengeID) const
{
    auto* challenge = m_challenges.get(challengeID);
    if (!challenge) {
        ASSERT_NOT_REACHED();
        return { };
    }

    Vector<AuthenticationChallengeIdentifier> challengesToCoalesce;
    challengesToCoalesce.append(challengeID);

    if (!canCoalesceChallenge(challenge->challenge))
        return challengesToCoalesce;

    for (auto& item : m_challenges) {
        if (item.key != challengeID && item.value->pageID == challenge->pageID && ProtectionSpace::compare(challenge->challenge.protectionSpace(), item.value->challenge.protectionSpace()))
            challengesToCoalesce.append(item.key);
    }

    return challengesToCoalesce;
}

void AuthenticationManager::didReceiveAuthenticationChallenge(PAL::SessionID sessionID, std::optional<WebPageProxyIdentifier> pageID, const SecurityOriginData* topOrigin, const AuthenticationChallenge& authenticationChallenge, NegotiatedLegacyTLS negotiatedLegacyTLS, ChallengeCompletionHandler&& completionHandler)
{
    if (!pageID)
        return completionHandler(AuthenticationChallengeDisposition::PerformDefaultHandling, { });

    auto challengeID = addChallengeToChallengeMap(makeUniqueRef<Challenge>(*pageID, authenticationChallenge, WTFMove(completionHandler)));

    // Coalesce challenges in the same protection space and in the same page.
    if (shouldCoalesceChallenge(*pageID, challengeID, authenticationChallenge))
        return;

    std::optional<SecurityOriginData> topOriginData;
    if (topOrigin)
        topOriginData = *topOrigin;
    protectedProcess()->send(Messages::NetworkProcessProxy::DidReceiveAuthenticationChallenge(sessionID, *pageID, topOriginData, authenticationChallenge, negotiatedLegacyTLS == NegotiatedLegacyTLS::Yes, challengeID));
}

void AuthenticationManager::didReceiveAuthenticationChallenge(IPC::MessageSender& download, const WebCore::AuthenticationChallenge& authenticationChallenge, ChallengeCompletionHandler&& completionHandler)
{
    std::optional<WebPageProxyIdentifier> dummyPageID;
    auto challengeID = addChallengeToChallengeMap(makeUniqueRef<Challenge>(dummyPageID, authenticationChallenge, WTFMove(completionHandler)));

    // Coalesce challenges in the same protection space and in the same page.
    if (shouldCoalesceChallenge(dummyPageID, challengeID, authenticationChallenge))
        return;

    download.send(Messages::DownloadProxy::DidReceiveAuthenticationChallenge(authenticationChallenge, challengeID));
}

void AuthenticationManager::completeAuthenticationChallenge(AuthenticationChallengeIdentifier challengeID, AuthenticationChallengeDisposition disposition, WebCore::Credential&& credential)
{
    ASSERT(RunLoop::isMain());

    for (auto& coalescedChallengeID : coalesceChallengesMatching(challengeID)) {
        auto challenge = m_challenges.take(coalescedChallengeID);
        ASSERT(!challenge->challenge.isNull());
        challenge->completionHandler(disposition, credential);
    }
}

void AuthenticationManager::negotiatedLegacyTLS(WebPageProxyIdentifier pageID) const
{
    protectedProcess()->send(Messages::NetworkProcessProxy::NegotiatedLegacyTLS(pageID));
}

} // namespace WebKit
