/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#include "NetworkProcess.h"

#include "NetworkCache.h"
#include "NetworkProcessCreationParameters.h"
#include "NetworkSessionCurl.h"
#include <WebCore/CurlContext.h>
#include <WebCore/NetworkStorageSession.h>
#include <WebCore/NotImplemented.h>
#include <wtf/CallbackAggregator.h>

namespace WebKit {

using namespace WebCore;

void NetworkProcess::platformInitializeNetworkProcess(const NetworkProcessCreationParameters&)
{
}

void NetworkProcess::allowSpecificHTTPSCertificateForHost(PAL::SessionID, const CertificateInfo& certificateInfo, const String& host)
{
    notImplemented();
}

void NetworkProcess::clearDiskCache(WallTime modifiedSince, CompletionHandler<void()>&& completionHandler)
{
    auto aggregator = CallbackAggregator::create(WTFMove(completionHandler));
    forEachNetworkSession([modifiedSince, &aggregator](NetworkSession& session) {
        if (auto* cache = session.cache())
            cache->clear(modifiedSince, [aggregator] () { });
    });
}

void NetworkProcess::platformTerminate()
{
    notImplemented();
}

void NetworkProcess::setNetworkProxySettings(PAL::SessionID sessionID, WebCore::CurlProxySettings&& settings)
{
    if (auto* networkStorageSession = storageSession(sessionID))
        networkStorageSession->setProxySettings(settings);
    else
        ASSERT_NOT_REACHED();
}

} // namespace WebKit
