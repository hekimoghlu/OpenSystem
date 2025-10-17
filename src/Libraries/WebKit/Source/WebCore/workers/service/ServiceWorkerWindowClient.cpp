/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#include "ServiceWorkerWindowClient.h"

#include "JSDOMPromiseDeferred.h"
#include "JSServiceWorkerWindowClient.h"
#include "SWContextManager.h"
#include "ServiceWorkerClients.h"
#include "ServiceWorkerGlobalScope.h"
#include "ServiceWorkerThread.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

ServiceWorkerWindowClient::ServiceWorkerWindowClient(ServiceWorkerGlobalScope& context, ServiceWorkerClientData&& data)
    : ServiceWorkerClient(context, WTFMove(data))
{
}

void ServiceWorkerWindowClient::focus(ScriptExecutionContext& context, Ref<DeferredPromise>&& promise)
{
    auto& serviceWorkerContext = downcast<ServiceWorkerGlobalScope>(context);

    if (context.settingsValues().serviceWorkersUserGestureEnabled && !serviceWorkerContext.isProcessingUserGesture()) {
        promise->reject(Exception { ExceptionCode::InvalidAccessError, "WindowClient focus requires a user gesture"_s });
        return;
    }

    auto promiseIdentifier = serviceWorkerContext.clients().addPendingPromise(WTFMove(promise));
    callOnMainThread([clientIdentifier = identifier(), promiseIdentifier, serviceWorkerIdentifier = serviceWorkerContext.thread().identifier()]() mutable {
        SWContextManager::singleton().connection()->focus(clientIdentifier, [promiseIdentifier, serviceWorkerIdentifier](auto result) mutable {
            SWContextManager::singleton().postTaskToServiceWorker(serviceWorkerIdentifier, [promiseIdentifier, result = crossThreadCopy(WTFMove(result))](auto& serviceWorkerContext) mutable {
                auto promise = serviceWorkerContext.clients().takePendingPromise(promiseIdentifier);
                if (!promise)
                    return;

                // FIXME: Check isFocused state and reject if not focused.
                if (!result) {
                    promise->reject(Exception { ExceptionCode::TypeError, "WindowClient focus failed"_s });
                    return;
                }

                promise->template resolve<IDLInterface<ServiceWorkerWindowClient>>(ServiceWorkerWindowClient::create(serviceWorkerContext, WTFMove(*result)));
            });
        });
    });
}

void ServiceWorkerWindowClient::navigate(ScriptExecutionContext& context, const String& urlString, Ref<DeferredPromise>&& promise)
{
    auto url = context.completeURL(urlString);

    if (!url.isValid()) {
        promise->reject(Exception { ExceptionCode::TypeError, makeString("URL string "_s, urlString, " cannot successfully be parsed"_s) });
        return;
    }

    if (url.protocolIsAbout()) {
        promise->reject(Exception { ExceptionCode::TypeError, makeString("ServiceWorkerClients.navigate() cannot be called with URL "_s, url.string()) });
        return;
    }

    // We implement step 4 (checking of client's active service worker) in network process as we cannot do it synchronously.
    auto& serviceWorkerContext = downcast<ServiceWorkerGlobalScope>(context);
    auto promiseIdentifier = serviceWorkerContext.clients().addPendingPromise(WTFMove(promise));
    callOnMainThread([clientIdentifier = identifier(), promiseIdentifier, serviceWorkerIdentifier = serviceWorkerContext.thread().identifier(), url = WTFMove(url).isolatedCopy()]() mutable {
        SWContextManager::singleton().connection()->navigate(clientIdentifier, serviceWorkerIdentifier, url, [promiseIdentifier, serviceWorkerIdentifier](auto result) mutable {
            SWContextManager::singleton().postTaskToServiceWorker(serviceWorkerIdentifier, [promiseIdentifier, result = crossThreadCopy(WTFMove(result))](auto& serviceWorkerContext) mutable {
                auto promise = serviceWorkerContext.clients().takePendingPromise(promiseIdentifier);
                if (!promise)
                    return;

                if (result.hasException()) {
                    promise->reject(result.releaseException());
                    return;
                }
                auto clientData = result.releaseReturnValue();
                if (!clientData) {
                    promise->resolveWithJSValue(JSC::jsNull());
                    return;
                }
#if ASSERT_ENABLED
                auto originData = SecurityOriginData::fromURL(clientData->url);
                ClientOrigin clientOrigin { originData, originData };
#endif
                ASSERT(serviceWorkerContext.clientOrigin() == clientOrigin);
                promise->template resolve<IDLInterface<ServiceWorkerWindowClient>>(ServiceWorkerWindowClient::create(serviceWorkerContext, WTFMove(*clientData)));
            });
        });
    });
}

} // namespace WebCore
