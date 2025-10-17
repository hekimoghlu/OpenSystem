/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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
#include "WebSocketProvider.h"

#include "NetworkProcessConnection.h"
#include "WebProcess.h"
#include "WebSocketChannelManager.h"
#include "WebTransportSession.h"
#include <WebCore/DocumentInlines.h>
#include <WebCore/WebTransportSessionClient.h>
#include <WebCore/WorkerGlobalScope.h>

namespace WebKit {
using namespace WebCore;

RefPtr<ThreadableWebSocketChannel> WebSocketProvider::createWebSocketChannel(Document& document, WebSocketChannelClient& client)
{
    return WebKit::WebSocketChannel::create(m_webPageProxyID, document, client);
}

Ref<WebCore::WebTransportSessionPromise> WebSocketProvider::initializeWebTransportSession(ScriptExecutionContext& context, WebTransportSessionClient& client, const URL& url)
{
    if (RefPtr scope = dynamicDowncast<WorkerGlobalScope>(context)) {
        ASSERT(!RunLoop::isMain());
        WebCore::WebTransportSessionPromise::Producer producer;
        Ref<WebCore::WebTransportSessionPromise> promise = producer.promise();

        RunLoop::protectedMain()->dispatch([
            contextID = context.identifier(),
            producer = WTFMove(producer),
            webPageProxyID = m_webPageProxyID,
            origin = crossThreadCopy(scope->clientOrigin()),
            client = ThreadSafeWeakPtr { client },
            url = crossThreadCopy(url)
        ] mutable {
            WebKit::WebTransportSession::initialize(WebProcess::singleton().ensureNetworkProcessConnection().connection(), WTFMove(client), url, webPageProxyID, origin)->whenSettled(RunLoop::protectedMain(), [producer = WTFMove(producer)] (auto&& result) mutable {
                if (!result)
                    producer.reject();
                else
                    producer.resolve(WTFMove(*result));
            });
        });
        return promise;
    }

    Ref document = downcast<Document>(context);
    ASSERT(RunLoop::isMain());
    return WebKit::WebTransportSession::initialize(WebProcess::singleton().ensureNetworkProcessConnection().connection(), client, url, m_webPageProxyID, document->clientOrigin());
}

} // namespace WebKit
