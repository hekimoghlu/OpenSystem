/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 13, 2023.
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
#include "WindowOrWorkerGlobalScopeFetch.h"

#include "CachedResourceRequestInitiatorTypes.h"
#include "Document.h"
#include "EventLoop.h"
#include "FetchResponse.h"
#include "JSDOMPromiseDeferred.h"
#include "JSFetchResponse.h"
#include "LocalDOMWindow.h"
#include "UserGestureIndicator.h"
#include "WorkerGlobalScope.h"

namespace WebCore {

using FetchResponsePromise = DOMPromiseDeferred<IDLInterface<FetchResponse>>;

// https://fetch.spec.whatwg.org/#dom-global-fetch
static void doFetch(ScriptExecutionContext& scope, FetchRequest::Info&& input, FetchRequest::Init&& init, FetchResponsePromise&& promise)
{
    auto requestOrException = FetchRequest::create(scope, WTFMove(input), WTFMove(init));
    if (requestOrException.hasException()) {
        promise.reject(requestOrException.releaseException());
        return;
    }

    auto request = requestOrException.releaseReturnValue();
    if (request->signal().aborted()) {
        auto reason = request->signal().reason().getValue();
        if (reason.isUndefined())
            promise.reject(Exception { ExceptionCode::AbortError, "Request signal is aborted"_s });
        else
            promise.rejectType<IDLAny>(reason);

        return;
    }

    FetchResponse::fetch(scope, request.get(), [promise = WTFMove(promise), scope = Ref { scope }, userGestureToken = UserGestureIndicator::currentUserGesture()]<typename Result> (Result&& result) mutable {
        scope->eventLoop().queueTask(TaskSource::Networking, [promise = WTFMove(promise), userGestureToken = WTFMove(userGestureToken), result = std::forward<Result>(result)] () mutable {
            if (!userGestureToken || userGestureToken->hasExpired(UserGestureToken::maximumIntervalForUserGestureForwardingForFetch()) || !userGestureToken->processingUserGesture()) {
                promise.settle(WTFMove(result));
                return;
            }
            UserGestureIndicator gestureIndicator(userGestureToken, UserGestureToken::GestureScope::MediaOnly, UserGestureToken::IsPropagatedFromFetch::Yes);
            promise.settle(WTFMove(result));
        });
    }, cachedResourceRequestInitiatorTypes().fetch);
}

void WindowOrWorkerGlobalScopeFetch::fetch(DOMWindow& window, FetchRequest::Info&& input, FetchRequest::Init&& init, Ref<DeferredPromise>&& promise)
{
    RefPtr localWindow = dynamicDowncast<LocalDOMWindow>(window);
    if (!localWindow) {
        promise->reject(ExceptionCode::InvalidStateError);
        return;
    }
    RefPtr document = localWindow->document();
    if (!document) {
        promise->reject(ExceptionCode::InvalidStateError);
        return;
    }
    doFetch(*document, WTFMove(input), WTFMove(init), WTFMove(promise));
}

void WindowOrWorkerGlobalScopeFetch::fetch(WorkerGlobalScope& scope, FetchRequest::Info&& input, FetchRequest::Init&& init, Ref<DeferredPromise>&& promise)
{
    doFetch(scope, WTFMove(input), WTFMove(init), WTFMove(promise));
}

}
