/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 2, 2023.
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
#include "WebsiteDataStore.h"

#include "APIHTTPCookieStore.h"
#include "NetworkProcessMessages.h"
#include "WebProcessPool.h"
#include "WebsiteDataStoreParameters.h"

namespace WebKit {

void WebsiteDataStore::platformSetNetworkParameters(WebsiteDataStoreParameters& parameters)
{
    auto& networkSessionParameters = parameters.networkSessionParameters;
    networkSessionParameters.persistentCredentialStorageEnabled = m_persistentCredentialStorageEnabled;
    networkSessionParameters.ignoreTLSErrors = m_ignoreTLSErrors;
    networkSessionParameters.proxySettings = m_networkProxySettings;
    networkSessionParameters.cookiePersistentStoragePath = m_cookiePersistentStoragePath;
    networkSessionParameters.cookiePersistentStorageType = m_cookiePersistentStorageType;
    networkSessionParameters.cookieAcceptPolicy = m_cookieAcceptPolicy;
}

void WebsiteDataStore::setPersistentCredentialStorageEnabled(bool enabled)
{
    if (persistentCredentialStorageEnabled() == enabled)
        return;

    if (enabled && !isPersistent())
        return;

    m_persistentCredentialStorageEnabled = enabled;
    networkProcess().send(Messages::NetworkProcess::SetPersistentCredentialStorageEnabled(m_sessionID, m_persistentCredentialStorageEnabled), 0);
}

void WebsiteDataStore::setIgnoreTLSErrors(bool ignoreTLSErrors)
{
    if (m_ignoreTLSErrors == ignoreTLSErrors)
        return;

    m_ignoreTLSErrors = ignoreTLSErrors;
    networkProcess().send(Messages::NetworkProcess::SetIgnoreTLSErrors(m_sessionID, m_ignoreTLSErrors), 0);
}

void WebsiteDataStore::setNetworkProxySettings(WebCore::SoupNetworkProxySettings&& settings)
{
    m_networkProxySettings = WTFMove(settings);
    networkProcess().send(Messages::NetworkProcess::SetNetworkProxySettings(m_sessionID, m_networkProxySettings), 0);
}

void WebsiteDataStore::setCookiePersistentStorage(const String& storagePath, SoupCookiePersistentStorageType storageType)
{
    if (m_cookiePersistentStoragePath == storagePath && m_cookiePersistentStorageType == storageType)
        return;

    m_cookiePersistentStoragePath = storagePath;
    m_cookiePersistentStorageType = storageType;
    cookieStore().setCookiePersistentStorage(m_cookiePersistentStoragePath, m_cookiePersistentStorageType);
}

void WebsiteDataStore::setHTTPCookieAcceptPolicy(WebCore::HTTPCookieAcceptPolicy policy)
{
    if (m_cookieAcceptPolicy == policy)
        return;

    m_cookieAcceptPolicy = policy;
    cookieStore().setHTTPCookieAcceptPolicy(policy, [] { });
}

} // namespace WebKit
