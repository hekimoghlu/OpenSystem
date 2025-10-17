/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include "WebCookieManager.h"

#include "Logging.h"
#include "MessageSenderInlines.h"
#include "NetworkProcess.h"
#include "NetworkProcessProxyMessages.h"
#include "WebCookieManagerMessages.h"
#include <WebCore/Cookie.h>
#include <WebCore/CookieStorage.h>
#include <WebCore/HTTPCookieAcceptPolicy.h>
#include <WebCore/NetworkStorageSession.h>
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/URL.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebCookieManager);

ASCIILiteral WebCookieManager::supplementName()
{
    return "WebCookieManager"_s;
}

WebCookieManager::WebCookieManager(NetworkProcess& process)
    : m_process(process)
{
    process.addMessageReceiver(Messages::WebCookieManager::messageReceiverName(), *this);
}

WebCookieManager::~WebCookieManager() = default;

void WebCookieManager::ref() const
{
    m_process->ref();
}

void WebCookieManager::deref() const
{
    m_process->deref();
}

Ref<NetworkProcess> WebCookieManager::protectedProcess()
{
    ASSERT(RunLoop::isMain());
    return m_process.get();
}

void WebCookieManager::getHostnamesWithCookies(PAL::SessionID sessionID, CompletionHandler<void(Vector<String>&&)>&& completionHandler)
{
    HashSet<String> hostnames;
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        storageSession->getHostnamesWithCookies(hostnames);
    completionHandler(copyToVector(hostnames));
}

void WebCookieManager::deleteCookiesForHostnames(PAL::SessionID sessionID, const Vector<String>& hostnames, CompletionHandler<void()>&& completionHandler)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        storageSession->deleteCookiesForHostnames(hostnames, WTFMove(completionHandler));
    else
        completionHandler();
}

void WebCookieManager::deleteAllCookies(PAL::SessionID sessionID, CompletionHandler<void()>&& completionHandler)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        storageSession->deleteAllCookies(WTFMove(completionHandler));
    else
        completionHandler();
}

void WebCookieManager::deleteCookie(PAL::SessionID sessionID, const Cookie& cookie, CompletionHandler<void()>&& completionHandler)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        storageSession->deleteCookie(cookie, WTFMove(completionHandler));
    else
        completionHandler();
}

void WebCookieManager::deleteAllCookiesModifiedSince(PAL::SessionID sessionID, WallTime time, CompletionHandler<void()>&& completionHandler)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        storageSession->deleteAllCookiesModifiedSince(time, WTFMove(completionHandler));
    else
        completionHandler();
}

void WebCookieManager::getAllCookies(PAL::SessionID sessionID, CompletionHandler<void(Vector<WebCore::Cookie>&&)>&& completionHandler)
{
    Vector<Cookie> cookies;
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        cookies = storageSession->getAllCookies();
    completionHandler(WTFMove(cookies));
}

void WebCookieManager::getCookies(PAL::SessionID sessionID, const URL& url, CompletionHandler<void(Vector<WebCore::Cookie>&&)>&& completionHandler)
{
    Vector<Cookie> cookies;
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        cookies = storageSession->getCookies(url);
    completionHandler(WTFMove(cookies));
}

void WebCookieManager::setCookie(PAL::SessionID sessionID, const Vector<Cookie>& cookies, CompletionHandler<void()>&& completionHandler)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID)) {
        for (auto& cookie : cookies)
            storageSession->setCookie(cookie);
    }
    completionHandler();
}

void WebCookieManager::setCookies(PAL::SessionID sessionID, const Vector<Cookie>& cookies, const URL& url, const URL& mainDocumentURL, CompletionHandler<void()>&& completionHandler)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        storageSession->setCookies(cookies, url, mainDocumentURL);
    completionHandler();
}

void WebCookieManager::notifyCookiesDidChange(PAL::SessionID sessionID)
{
    ASSERT(RunLoop::isMain());
    protectedProcess()->send(Messages::NetworkProcessProxy::CookiesDidChange(sessionID), 0);
}

void WebCookieManager::startObservingCookieChanges(PAL::SessionID sessionID)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID)) {
        WebCore::startObservingCookieChanges(*storageSession, [weakThis = WeakPtr { *this }, sessionID] {
            if (RefPtr protectedThis = weakThis.get())
                protectedThis->notifyCookiesDidChange(sessionID);
        });
    }
}

void WebCookieManager::stopObservingCookieChanges(PAL::SessionID sessionID)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        WebCore::stopObservingCookieChanges(*storageSession);
}

void WebCookieManager::setHTTPCookieAcceptPolicy(PAL::SessionID sessionID, HTTPCookieAcceptPolicy policy, CompletionHandler<void()>&& completionHandler)
{
    RELEASE_LOG(Storage, "WebCookieManager::setHTTPCookieAcceptPolicy set policy %d for session %" PRIu64, static_cast<int>(policy), sessionID.toUInt64());
    platformSetHTTPCookieAcceptPolicy(sessionID, policy, [policy, process = protectedProcess(), completionHandler = WTFMove(completionHandler)] () mutable {
        process->cookieAcceptPolicyChanged(policy);
        completionHandler();
    });
}

void WebCookieManager::getHTTPCookieAcceptPolicy(PAL::SessionID sessionID, CompletionHandler<void(HTTPCookieAcceptPolicy)>&& completionHandler)
{
    if (auto* storageSession = protectedProcess()->storageSession(sessionID))
        completionHandler(storageSession->cookieAcceptPolicy());
    else
        completionHandler(HTTPCookieAcceptPolicy::Never);
}

} // namespace WebKit
