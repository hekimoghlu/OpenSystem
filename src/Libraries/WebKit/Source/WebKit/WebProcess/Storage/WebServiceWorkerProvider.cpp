/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 16, 2024.
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
#include "WebServiceWorkerProvider.h"

#include "NetworkProcessConnection.h"
#include "WebProcess.h"
#include "WebSWClientConnection.h"
#include "WebSWServerConnection.h"
#include <WebCore/CachedResource.h>
#include <WebCore/Exception.h>
#include <WebCore/ExceptionCode.h>
#include <WebCore/LegacySchemeRegistry.h>
#include <WebCore/ServiceWorkerJob.h>
#include <wtf/text/WTFString.h>

namespace WebKit {
using namespace PAL;
using namespace WebCore;

WebServiceWorkerProvider& WebServiceWorkerProvider::singleton()
{
    static NeverDestroyed<WebServiceWorkerProvider> provider;
    return provider;
}

WebServiceWorkerProvider::WebServiceWorkerProvider()
{
}

WebCore::SWClientConnection& WebServiceWorkerProvider::serviceWorkerConnection()
{
    return WebProcess::singleton().ensureNetworkProcessConnection().serviceWorkerConnection();
}

WebCore::SWClientConnection* WebServiceWorkerProvider::existingServiceWorkerConnection()
{
    RefPtr networkProcessConnection = WebProcess::singleton().existingNetworkProcessConnection();
    if (!networkProcessConnection)
        return nullptr;

    return &networkProcessConnection->serviceWorkerConnection();
}

void WebServiceWorkerProvider::updateThrottleState(bool isThrottleable)
{
    auto* networkProcessConnection = WebProcess::singleton().existingNetworkProcessConnection();
    if (!networkProcessConnection)
        return;
    auto& connection = networkProcessConnection->serviceWorkerConnection();
    if (isThrottleable != connection.isThrottleable())
        connection.updateThrottleState();
}

void WebServiceWorkerProvider::terminateWorkerForTesting(WebCore::ServiceWorkerIdentifier identifier, CompletionHandler<void()>&& callback)
{
    WebProcess::singleton().ensureNetworkProcessConnection().serviceWorkerConnection().terminateWorkerForTesting(identifier, WTFMove(callback));
}

} // namespace WebKit
