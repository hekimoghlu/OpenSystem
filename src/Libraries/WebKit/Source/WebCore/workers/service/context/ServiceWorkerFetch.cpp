/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 1, 2024.
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
#include "ServiceWorkerFetch.h"

#include "CrossOriginAccessControl.h"
#include "EventNames.h"
#include "FetchEvent.h"
#include "FetchRequest.h"
#include "FetchResponse.h"
#include "JSDOMPromise.h"
#include "JSDOMPromiseDeferred.h"
#include "MIMETypeRegistry.h"
#include "ResourceRequest.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorker.h"
#include "ServiceWorkerGlobalScope.h"
#include "ServiceWorkerThread.h"
#include "WorkerGlobalScope.h"

namespace WebCore {

namespace ServiceWorkerFetch {

// https://fetch.spec.whatwg.org/#http-fetch step 5.5
static inline ResourceError validateResponse(const ResourceResponse& response, FetchOptions::Mode mode, FetchOptions::Redirect redirect)
{
    if (response.type() == ResourceResponse::Type::Error)
        return ResourceError { errorDomainWebKitInternal, 0, response.url(), "Response served by service worker is an error"_s, ResourceError::Type::General, ResourceError::IsSanitized::Yes };

    if (mode == FetchOptions::Mode::SameOrigin && response.type() == ResourceResponse::Type::Cors)
        return ResourceError { errorDomainWebKitInternal, 0, response.url(), "Response served by service worker is CORS while mode is same origin"_s, ResourceError::Type::AccessControl, ResourceError::IsSanitized::Yes };

    if (mode != FetchOptions::Mode::NoCors && response.tainting() == ResourceResponse::Tainting::Opaque)
        return ResourceError { errorDomainWebKitInternal, 0, response.url(), "Response served by service worker is opaque"_s, ResourceError::Type::AccessControl, ResourceError::IsSanitized::Yes };

    // Navigate mode induces manual redirect.
    if (redirect != FetchOptions::Redirect::Manual && mode != FetchOptions::Mode::Navigate && response.tainting() == ResourceResponse::Tainting::Opaqueredirect)
        return ResourceError { errorDomainWebKitInternal, 0, response.url(), "Response served by service worker is opaque redirect"_s, ResourceError::Type::AccessControl, ResourceError::IsSanitized::Yes };

    if ((redirect != FetchOptions::Redirect::Follow || mode == FetchOptions::Mode::Navigate) && response.isRedirected())
        return ResourceError { errorDomainWebKitInternal, 0, response.url(), "Response served by service worker has redirections"_s, ResourceError::Type::AccessControl, ResourceError::IsSanitized::Yes };

    return { };
}

static void processResponse(Ref<Client>&& client, Expected<Ref<FetchResponse>, std::optional<ResourceError>>&& result, FetchOptions::Mode mode, FetchOptions::Redirect redirect, const URL& requestURL, CertificateInfo&& certificateInfo, DeferredPromise& promise)
{
    if (!result.has_value()) {
        auto& error = result.error();
        if (!error) {
            client->didNotHandle();
            promise.resolve();
            return;
        }
        client->didFail(*error);
        promise.reject(Exception { ExceptionCode::NetworkError });
        return;
    }
    auto response = WTFMove(result.value());

    auto loadingError = response->loadingError();
    if (!loadingError.isNull()) {
        client->didFail(loadingError);
        promise.reject(Exception { ExceptionCode::NetworkError });
        return;
    }

    auto resourceResponse = response->resourceResponse();
    if (auto error = validateResponse(resourceResponse, mode, redirect); !error.isNull()) {
        client->didFail(error);
        promise.reject(Exception { ExceptionCode::NetworkError });
        return;
    }

    promise.resolve();

    if (response->isAvailableNavigationPreload()) {
        client->usePreload();
        response->markAsUsedForPreload();
        return;
    }

    // As per https://fetch.spec.whatwg.org/#main-fetch step 9, copy request's url list in response's url list if empty.
    if (resourceResponse.url().isNull())
        resourceResponse.setURL(requestURL);

    if (resourceResponse.isRedirection() && resourceResponse.httpHeaderFields().contains(HTTPHeaderName::Location)) {
        client->didReceiveRedirection(resourceResponse);
        return;
    }

    // In case of main resource and mime type is the default one, we set it to text/html to pass more service worker WPT tests.
    // FIXME: We should refine our MIME type sniffing strategy for synthetic responses.
    if (mode == FetchOptions::Mode::Navigate) {
        if (resourceResponse.mimeType() == defaultMIMEType() && !resourceResponse.isNosniff()) {
            resourceResponse.setMimeType("text/html"_s);
            resourceResponse.setTextEncodingName("UTF-8"_s);
        }

        if (!resourceResponse.certificateInfo())
            resourceResponse.setCertificateInfo(WTFMove(certificateInfo));
    }

    client->didReceiveResponse(resourceResponse);

    if (response->isBodyReceivedByChunk()) {
        client->setCancelledCallback([response = WeakPtr { response.get() }] {
            if (RefPtr protectedResponse = response.get())
                protectedResponse->cancelStream();
        });
        response->consumeBodyReceivedByChunk([client = WTFMove(client), response = WeakPtr { response.get() }] (auto&& result) mutable {
            if (result.hasException()) {
                auto error = FetchEvent::createResponseError(URL { }, result.exception().message(), ResourceError::IsSanitized::Yes);
                client->didFail(error);
                return;
            }

            if (auto* chunk = result.returnValue())
                client->didReceiveData(SharedBuffer::create(*chunk));
            else
                client->didFinish(response ? response->networkLoadMetrics() : NetworkLoadMetrics { });
        });
        return;
    }

    auto body = response->consumeBody();
    WTF::switchOn(body, [&] (Ref<FormData>& formData) {
        client->didReceiveFormDataAndFinish(WTFMove(formData));
    }, [&] (Ref<SharedBuffer>& buffer) {
        client->didReceiveData(WTFMove(buffer));
        client->didFinish(response->networkLoadMetrics());
    }, [&] (std::nullptr_t&) {
        client->didFinish(response->networkLoadMetrics());
    });
}

void dispatchFetchEvent(Ref<Client>&& client, ServiceWorkerGlobalScope& globalScope, ResourceRequest&& request, String&& referrer, FetchOptions&& options, SWServerConnectionIdentifier connectionIdentifier, FetchIdentifier fetchIdentifier, bool isServiceWorkerNavigationPreloadEnabled, String&& clientIdentifier, String&& resultingClientIdentifier)
{
    auto requestHeaders = FetchHeaders::create(FetchHeaders::Guard::Immutable, HTTPHeaderMap { request.httpHeaderFields() });

    FetchOptions::Mode mode = options.mode;
    FetchOptions::Redirect redirect = options.redirect;

    bool isNavigation = options.mode == FetchOptions::Mode::Navigate;

    ASSERT(globalScope.registration().active());
    ASSERT(globalScope.registration().active()->identifier() == globalScope.thread().identifier());
    // FIXME: we should use the same path for registration changes as for fetch events.
    ASSERT(globalScope.registration().active()->state() == ServiceWorkerState::Activated || globalScope.registration().active()->state() == ServiceWorkerState::Activating);

    auto formData = request.httpBody();
    std::optional<FetchBody> body;
    if (formData && !formData->isEmpty()) {
        body = FetchBody::fromFormData(globalScope, formData.releaseNonNull());
        if (!body) {
            client->didNotHandle();
            return;
        }
    }
    // FIXME: loading code should set redirect mode to manual.
    if (isNavigation)
        options.redirect = FetchOptions::Redirect::Manual;

    URL requestURL = request.url();
    auto fetchRequest = FetchRequest::create(globalScope, WTFMove(body), WTFMove(requestHeaders),  WTFMove(request), WTFMove(options), WTFMove(referrer));

    // The request has already passed content extension checks, no need to reapply them if service worker does the fetch itself.
    fetchRequest->disableContentExtensionsCheck();

    // If service worker navigation preload is not enabled, we do not want to reuse any preload directly.
    if (!isServiceWorkerNavigationPreloadEnabled)
        fetchRequest->setNavigationPreloadIdentifier(fetchIdentifier);

    FetchEvent::Init init;
    init.request = WTFMove(fetchRequest);
    init.resultingClientId = WTFMove(resultingClientIdentifier);
    init.clientId = WTFMove(clientIdentifier);
    init.cancelable = true;

    auto& jsDOMGlobalObject = *JSC::jsCast<JSDOMGlobalObject*>(globalScope.globalObject());
    JSC::JSLockHolder lock(jsDOMGlobalObject.vm());

    auto* promise = JSC::JSPromise::create(jsDOMGlobalObject.vm(), jsDOMGlobalObject.promiseStructure());
    ASSERT(promise);

    auto deferredPromise = DeferredPromise::create(jsDOMGlobalObject, *promise);
    init.handled = DOMPromise::create(jsDOMGlobalObject, *promise);

    auto event = FetchEvent::create(*globalScope.globalObject(), eventNames().fetchEvent, WTFMove(init), Event::IsTrusted::Yes);
    if (isServiceWorkerNavigationPreloadEnabled) {
        globalScope.addFetchEvent({ connectionIdentifier, fetchIdentifier }, event.get());
        event->setNavigationPreloadIdentifier(fetchIdentifier);
    }

    CertificateInfo certificateInfo = globalScope.certificateInfo();

    event->onResponse([client, mode, redirect, requestURL, certificateInfo = WTFMove(certificateInfo), deferredPromise]<typename Result> (Result&& result) mutable {
        processResponse(WTFMove(client), std::forward<Result>(result), mode, redirect, requestURL, WTFMove(certificateInfo), deferredPromise.get());
    });

    globalScope.dispatchEvent(event);

    if (!event->respondWithEntered()) {
        if (event->defaultPrevented()) {
            ResourceError error { errorDomainWebKitInternal, 0, requestURL, "Fetch event was canceled"_s, ResourceError::Type::General, ResourceError::IsSanitized::Yes };
            client->didFail(error);
            deferredPromise->reject(Exception { ExceptionCode::NetworkError });
            return;
        }
        client->didNotHandle();
        deferredPromise->resolve();
    }

    globalScope.updateExtendedEventsSet(event.ptr());
}

} // namespace ServiceWorkerFetch

} // namespace WebCore
