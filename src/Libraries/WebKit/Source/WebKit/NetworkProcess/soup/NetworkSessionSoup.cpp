/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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
#include "NetworkSessionSoup.h"

#include "NetworkProcess.h"
#include "NetworkSessionCreationParameters.h"
#include "WebCookieManager.h"
#include "WebSocketTaskSoup.h"
#include <WebCore/DeprecatedGlobalSettings.h>
#include <WebCore/NetworkStorageSession.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/SoupNetworkSession.h>
#include <libsoup/soup.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(NetworkSessionSoup);

NetworkSessionSoup::NetworkSessionSoup(NetworkProcess& networkProcess, const NetworkSessionCreationParameters& parameters)
    : NetworkSession(networkProcess, parameters)
    , m_networkSession(makeUnique<SoupNetworkSession>(m_sessionID))
    , m_persistentCredentialStorageEnabled(parameters.persistentCredentialStorageEnabled)
{
    auto* storageSession = networkStorageSession();
    ASSERT(storageSession);

    storageSession->setCookieAcceptPolicy(parameters.cookieAcceptPolicy);

    setIgnoreTLSErrors(parameters.ignoreTLSErrors);

    if (parameters.proxySettings.mode != SoupNetworkProxySettings::Mode::Default)
        setProxySettings(parameters.proxySettings);

    if (!parameters.cookiePersistentStoragePath.isEmpty())
        setCookiePersistentStorage(parameters.cookiePersistentStoragePath, parameters.cookiePersistentStorageType);
    else
        m_networkSession->setCookieJar(storageSession->cookieStorage());

    if (!parameters.hstsStorageDirectory.isEmpty())
        m_networkSession->setHSTSPersistentStorage(parameters.hstsStorageDirectory);
}

NetworkSessionSoup::~NetworkSessionSoup() = default;

SoupSession* NetworkSessionSoup::soupSession() const
{
    return m_networkSession->soupSession();
}

void NetworkSessionSoup::setCookiePersistentStorage(const String& storagePath, SoupCookiePersistentStorageType storageType)
{
    auto* storageSession = networkStorageSession();
    if (!storageSession)
        return;

    GRefPtr<SoupCookieJar> jar;
    switch (storageType) {
    case SoupCookiePersistentStorageType::Text:
        jar = adoptGRef(soup_cookie_jar_text_new(storagePath.utf8().data(), FALSE));
        break;
    case SoupCookiePersistentStorageType::SQLite:
        jar = adoptGRef(soup_cookie_jar_db_new(storagePath.utf8().data(), FALSE));
        break;
    }
    storageSession->setCookieStorage(WTFMove(jar));

    m_networkSession->setCookieJar(storageSession->cookieStorage());
}

void NetworkSessionSoup::clearCredentials(WallTime)
{
#if SOUP_CHECK_VERSION(2, 57, 1)
    soup_auth_manager_clear_cached_credentials(SOUP_AUTH_MANAGER(soup_session_get_feature(soupSession(), SOUP_TYPE_AUTH_MANAGER)));
#endif
}

#if USE(SOUP2)
static gboolean webSocketAcceptCertificateCallback(GTlsConnection* connection, GTlsCertificate* certificate, GTlsCertificateFlags errors, NetworkSessionSoup* session)
{
    if (DeprecatedGlobalSettings::allowsAnySSLCertificate())
        return TRUE;

    auto* soupMessage = static_cast<SoupMessage*>(g_object_get_data(G_OBJECT(connection), "wk-soup-message"));
    return !session->soupNetworkSession().checkTLSErrors(soupURIToURL(soup_message_get_uri(soupMessage)), certificate, errors);
}

static void webSocketMessageNetworkEventCallback(SoupMessage* soupMessage, GSocketClientEvent event, GIOStream* connection, NetworkSessionSoup* session)
{
    if (event != G_SOCKET_CLIENT_TLS_HANDSHAKING)
        return;

    g_object_set_data(G_OBJECT(connection), "wk-soup-message", soupMessage);
    g_signal_connect(connection, "accept-certificate", G_CALLBACK(webSocketAcceptCertificateCallback), session);
}
#endif

std::unique_ptr<WebSocketTask> NetworkSessionSoup::createWebSocketTask(WebPageProxyIdentifier webPageProxyID, std::optional<FrameIdentifier> frameID, std::optional<PageIdentifier> pageID, NetworkSocketChannel& channel, const ResourceRequest& request, const String& protocol, const ClientOrigin&, bool, bool, OptionSet<WebCore::AdvancedPrivacyProtections>, StoredCredentialsPolicy)
{
    GRefPtr<SoupMessage> soupMessage = request.createSoupMessage(blobRegistry());
    if (!soupMessage)
        return nullptr;

    if (request.url().protocolIs("wss"_s)) {
#if USE(SOUP2)
        g_signal_connect(soupMessage.get(), "network-event", G_CALLBACK(webSocketMessageNetworkEventCallback), this);
#else
        g_signal_connect(soupMessage.get(), "accept-certificate", G_CALLBACK(+[](SoupMessage* message, GTlsCertificate* certificate, GTlsCertificateFlags errors,  NetworkSessionSoup* session) -> gboolean {
            if (DeprecatedGlobalSettings::allowsAnySSLCertificate())
                return TRUE;

            return !session->soupNetworkSession().checkTLSErrors(soup_message_get_uri(message), certificate, errors);
        }), this);
#endif
    }

    bool shouldBlockCookies = networkStorageSession()->shouldBlockCookies(request, frameID, pageID, networkProcess().shouldRelaxThirdPartyCookieBlockingForPage(webPageProxyID));
    if (shouldBlockCookies)
        soup_message_disable_feature(soupMessage.get(), SOUP_TYPE_COOKIE_JAR);

    return makeUnique<WebSocketTask>(channel, request, soupSession(), soupMessage.get(), protocol);
}

void NetworkSessionSoup::setIgnoreTLSErrors(bool ignoreTLSErrors)
{
    m_networkSession->setIgnoreTLSErrors(ignoreTLSErrors);
}

void NetworkSessionSoup::allowSpecificHTTPSCertificateForHost(const CertificateInfo& certificateInfo, const String& host)
{
    m_networkSession->allowSpecificHTTPSCertificateForHost(certificateInfo, host);
}

void NetworkSessionSoup::setProxySettings(const SoupNetworkProxySettings& settings)
{
    m_networkSession->setProxySettings(settings);
}

} // namespace WebKit
