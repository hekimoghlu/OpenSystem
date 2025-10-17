/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#include "BackgroundFetchManager.h"

#include "BackgroundFetchInformation.h"
#include "BackgroundFetchRequest.h"
#include "ContentSecurityPolicy.h"
#include "FetchRequest.h"
#include "JSBackgroundFetchRegistration.h"
#include "SWClientConnection.h"
#include "ServiceWorkerProvider.h"
#include "ServiceWorkerRegistration.h"

namespace WebCore {

BackgroundFetchManager::BackgroundFetchManager(ServiceWorkerRegistration& registration)
    : m_identifier(registration.identifier())
{
}

BackgroundFetchManager::~BackgroundFetchManager()
{
}

static ExceptionOr<Vector<Ref<FetchRequest>>> buildBackgroundFetchRequests(ScriptExecutionContext& context, BackgroundFetchManager::Requests&& backgroundFetchRequests)
{
    return switchOn(WTFMove(backgroundFetchRequests), [&context] (RefPtr<FetchRequest>&& request) -> ExceptionOr<Vector<Ref<FetchRequest>>> {
        auto result = FetchRequest::create(context, request.releaseNonNull(), { });
        if (result.hasException())
            return result.releaseException();
        if (result.returnValue()->mode() == FetchOptions::Mode::NoCors)
            return Exception { ExceptionCode::TypeError, "Request has no-cors mode"_s };
        return Vector<Ref<FetchRequest>> { result.releaseReturnValue() };
    }, [&context] (String&& url) -> ExceptionOr<Vector<Ref<FetchRequest>>> {
        auto result = FetchRequest::create(context, WTFMove(url), { });
        if (result.hasException())
            return result.releaseException();
        return Vector<Ref<FetchRequest>> { result.releaseReturnValue() };
    }, [&context] (Vector<BackgroundFetchManager::RequestInfo>&& requestInfos) -> ExceptionOr<Vector<Ref<FetchRequest>>> {
        std::optional<Exception> exception;
        Vector<Ref<FetchRequest>> requests;
        requests.reserveInitialCapacity(requestInfos.size());
        for (auto& requestInfo : requestInfos) {
            auto result = FetchRequest::create(context, WTFMove(requestInfo), { });
            if (result.hasException())
                return result.releaseException();
            if (result.returnValue()->mode() == FetchOptions::Mode::NoCors)
                return Exception { ExceptionCode::TypeError, "Request has no-cors mode"_s };
            
            // FIXME: Add support for readable stream bodies
            if (result.returnValue()->isReadableStreamBody())
                return Exception { ExceptionCode::NotSupportedError, "ReadableStream uploading is not supported"_s };
            
            requests.append(result.releaseReturnValue());
        }
        return requests;
    });
}

Ref<BackgroundFetchRegistration> BackgroundFetchManager::backgroundFetchRegistrationInstance(ScriptExecutionContext& context, BackgroundFetchInformation&& data)
{
    auto identifier = data.identifier;
    auto result = m_backgroundFetchRegistrations.ensure(identifier, [&] {
        return BackgroundFetchRegistration::create(context, WTFMove(data));
    });

    auto registration = result.iterator->value;
    if (!result.isNewEntry)
        registration->updateInformation(data);
    return registration;
}

void BackgroundFetchManager::fetch(ScriptExecutionContext& context, const String& fetchIdentifier, Requests&& backgroundFetchRequests, BackgroundFetchOptions&& options, DOMPromiseDeferred<IDLInterface<BackgroundFetchRegistration>>&& promise)
{
    auto generatedRequests = buildBackgroundFetchRequests(context, WTFMove(backgroundFetchRequests));
    if (generatedRequests.hasException()) {
        promise.reject(generatedRequests.releaseException());
        return;
    }

    if (!generatedRequests.returnValue().size()) {
        promise.reject(Exception { ExceptionCode::TypeError, "No requests"_s });
        return;
    }

    auto requests = map(generatedRequests.releaseReturnValue(), [&](auto&& fetchRequest) -> BackgroundFetchRequest {
        Markable<ContentSecurityPolicyResponseHeaders, ContentSecurityPolicyResponseHeaders::MarkableTraits> responseHeaders;
        if (!context.shouldBypassMainWorldContentSecurityPolicy()) {
            if (CheckedPtr policy = context.contentSecurityPolicy())
                responseHeaders = policy->responseHeaders();
        }
        return { fetchRequest->resourceRequest(), fetchRequest->fetchOptions(), fetchRequest->headers().guard(), fetchRequest->headers().internalHeaders(), fetchRequest->internalRequestReferrer(), WTFMove(responseHeaders) };
    });
    SWClientConnection::fromScriptExecutionContext(context)->startBackgroundFetch(m_identifier, fetchIdentifier, WTFMove(requests), WTFMove(options), [weakThis = WeakPtr { *this }, weakContext = WeakPtr { context }, promise = WTFMove(promise)](ExceptionOr<std::optional<BackgroundFetchInformation>>&& result) mutable {
        if (!weakContext)
            return;
        weakContext->postTask([weakThis = WTFMove(weakThis), promise = WTFMove(promise), result = WTFMove(result)](auto& context) mutable {
            if (!weakThis)
                return;

            if (result.hasException()) {
                promise.reject(result.releaseException());
                return;
            }
            if (!result.returnValue()) {
                promise.reject(Exception { ExceptionCode::TypeError, "An internal error occured"_s });
                return;
            }

            promise.resolve(weakThis->backgroundFetchRegistrationInstance(context, *result.releaseReturnValue()));
        });

    });
}

void BackgroundFetchManager::get(ScriptExecutionContext& context, const String& fetchIdentifier, DOMPromiseDeferred<IDLNullable<IDLInterface<BackgroundFetchRegistration>>>&& promise)
{
    auto iterator = m_backgroundFetchRegistrations.find(fetchIdentifier);
    if (iterator == m_backgroundFetchRegistrations.end()) {
        promise.resolve(nullptr);
        return;
    }

    SWClientConnection::fromScriptExecutionContext(context)->backgroundFetchInformation(m_identifier, fetchIdentifier, [weakThis = WeakPtr { *this }, weakContext = WeakPtr { context }, promise = WTFMove(promise)](auto&& result) mutable {
        if (!weakContext)
            return;
        weakContext->postTask([weakThis = WTFMove(weakThis), promise = WTFMove(promise), result = WTFMove(result)](auto& context) mutable {
            if (!weakThis)
                return;

            if (result.hasException()) {
                promise.reject(result.releaseException());
                return;
            }

            RefPtr<BackgroundFetchRegistration> backgroundFetchRegistration;
            if (result.returnValue())
                backgroundFetchRegistration = weakThis->backgroundFetchRegistrationInstance(context, *result.releaseReturnValue());

            promise.resolve(backgroundFetchRegistration.get());
        });
    });
}

void BackgroundFetchManager::getIds(ScriptExecutionContext& context, DOMPromiseDeferred<IDLSequence<IDLDOMString>>&& promise)
{
    SWClientConnection::fromScriptExecutionContext(context)->backgroundFetchIdentifiers(m_identifier, [promise = WTFMove(promise)](Vector<String>&& result) mutable {
        promise.resolve(WTFMove(result));
    });
}

} // namespace WebCore
