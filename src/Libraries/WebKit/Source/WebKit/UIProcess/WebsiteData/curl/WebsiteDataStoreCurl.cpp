/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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

#include "NetworkProcessMessages.h"
#include "WebProcessPool.h"
#include "WebsiteDataStoreParameters.h"

namespace WebKit {

void WebsiteDataStore::platformSetNetworkParameters(WebsiteDataStoreParameters& parameters)
{
    auto& directories = resolvedDirectories();
    auto alternativeServiceStorageDirectory = directories.alternativeServicesDirectory;
    SandboxExtension::Handle alternativeServiceStorageDirectoryExtensionHandle;
    createHandleFromResolvedPathIfPossible(alternativeServiceStorageDirectory, alternativeServiceStorageDirectoryExtensionHandle);

    parameters.networkSessionParameters.alternativeServiceDirectory = WTFMove(alternativeServiceStorageDirectory);
    parameters.networkSessionParameters.alternativeServiceDirectoryExtensionHandle = WTFMove(alternativeServiceStorageDirectoryExtensionHandle);
    parameters.networkSessionParameters.cookiePersistentStorageFile = directories.cookieStorageFile;
    parameters.networkSessionParameters.proxySettings = m_proxySettings;
}

void WebsiteDataStore::setNetworkProxySettings(WebCore::CurlProxySettings&& proxySettings)
{
    m_proxySettings = WTFMove(proxySettings);

    networkProcess().send(Messages::NetworkProcess::SetNetworkProxySettings(m_sessionID, m_proxySettings), 0);
}

}
