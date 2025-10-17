/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#include "ServiceWorkerInternals.h"

#include "FetchEvent.h"
#include "FetchRequest.h"
#include "JSDOMPromiseDeferred.h"
#include "JSFetchResponse.h"
#include "NotificationPayload.h"
#include "PushSubscription.h"
#include "PushSubscriptionData.h"
#include "SWContextManager.h"
#include "ServiceWorkerClient.h"
#include "ServiceWorkerGlobalScope.h"
#include "ServiceWorkerRegistration.h"
#include <wtf/ProcessID.h>

namespace WebCore {

ServiceWorkerInternals::ServiceWorkerInternals(ServiceWorkerGlobalScope& globalScope, ServiceWorkerIdentifier identifier)
    : m_identifier(identifier)
{
    globalScope.setIsProcessingUserGestureForTesting(true);
}

ServiceWorkerInternals::~ServiceWorkerInternals() = default;

void ServiceWorkerInternals::setOnline(bool isOnline)
{
    callOnMainThread([identifier = m_identifier, isOnline] () {
        if (auto* proxy = SWContextManager::singleton().serviceWorkerThreadProxy(identifier))
            proxy->notifyNetworkStateChange(isOnline);
    });
}

void ServiceWorkerInternals::terminate()
{
    callOnMainThread([identifier = m_identifier] () {
        SWContextManager::singleton().terminateWorker(identifier, Seconds::infinity(), [] { });
    });
}

void ServiceWorkerInternals::schedulePushEvent(const String& message, RefPtr<DeferredPromise>&& promise)
{
    auto counter = ++m_pushEventCounter;
    m_pushEventPromises.add(counter, WTFMove(promise));

    std::optional<Vector<uint8_t>> data;
    if (!message.isNull()) {
        auto utf8 = message.utf8();
        data = Vector(utf8.span());
    }
    callOnMainThread([identifier = m_identifier, data = WTFMove(data), weakThis = WeakPtr { *this }, counter]() mutable {
        SWContextManager::singleton().firePushEvent(identifier, WTFMove(data), std::nullopt, [identifier, weakThis = WTFMove(weakThis), counter](bool result, std::optional<NotificationPayload>&&) mutable {
            if (auto* proxy = SWContextManager::singleton().serviceWorkerThreadProxy(identifier)) {
                proxy->thread().runLoop().postTaskForMode([weakThis = WTFMove(weakThis), counter, result](auto&) {
                    if (!weakThis)
                        return;
                    if (auto promise = weakThis->m_pushEventPromises.take(counter))
                        promise->resolve<IDLBoolean>(result);
                }, WorkerRunLoop::defaultMode());
            }
        });
    });
}

void ServiceWorkerInternals::schedulePushSubscriptionChangeEvent(PushSubscription* newSubscription, PushSubscription* oldSubscription)
{
    std::optional<PushSubscriptionData> newSubscriptionData;
    std::optional<PushSubscriptionData> oldSubscriptionData;

    if (newSubscription)
        newSubscriptionData = newSubscription->data().isolatedCopy();
    if (oldSubscription)
        oldSubscriptionData = oldSubscription->data().isolatedCopy();

    callOnMainThread([identifier = m_identifier, newSubscriptionData = WTFMove(newSubscriptionData), oldSubscriptionData = WTFMove(oldSubscriptionData)]() mutable {
        SWContextManager::singleton().firePushSubscriptionChangeEvent(identifier, WTFMove(newSubscriptionData), WTFMove(oldSubscriptionData));
    });
}

void ServiceWorkerInternals::waitForFetchEventToFinish(FetchEvent& event, DOMPromiseDeferred<IDLInterface<FetchResponse>>&& promise)
{
    event.onResponse([promise = WTFMove(promise), event = Ref { event }] (auto&& result) mutable {
        if (!result.has_value()) {
            String description;
            if (auto& error = result.error())
                description = error->localizedDescription();
            promise.reject(ExceptionCode::TypeError, description);
            return;
        }
        promise.resolve(WTFMove(result.value()));
    });
}

Ref<FetchEvent> ServiceWorkerInternals::createBeingDispatchedFetchEvent(ScriptExecutionContext& context)
{
    auto event = FetchEvent::createForTesting(context);
    event->setEventPhase(Event::CAPTURING_PHASE);
    return event;
}

Ref<FetchResponse> ServiceWorkerInternals::createOpaqueWithBlobBodyResponse(ScriptExecutionContext& context)
{
    auto blob = Blob::create(&context);
    auto formData = FormData::create();
    formData->appendBlob(blob->url());

    ResourceResponse response;
    response.setType(ResourceResponse::Type::Cors);
    response.setTainting(ResourceResponse::Tainting::Opaque);
    auto fetchResponse = FetchResponse::create(&context, FetchBody::fromFormData(context, WTFMove(formData)), FetchHeaders::Guard::Response, WTFMove(response));
    fetchResponse->initializeOpaqueLoadIdentifierForTesting();
    return fetchResponse;
}

Vector<String> ServiceWorkerInternals::fetchResponseHeaderList(FetchResponse& response)
{
    return WTF::map(response.internalResponseHeaders(), [](auto& keyValue) {
        return keyValue.key;
    });
}

#if !PLATFORM(MAC)
String ServiceWorkerInternals::processName() const
{
    return "none"_s;
}
#endif

bool ServiceWorkerInternals::isThrottleable() const
{
    auto* connection = SWContextManager::singleton().connection();
    return connection ? connection->isThrottleable() : true;
}

int ServiceWorkerInternals::processIdentifier() const
{
    return getCurrentProcessID();
}

void ServiceWorkerInternals::lastNavigationWasAppInitiated(Ref<DeferredPromise>&& promise)
{
    ASSERT(!m_lastNavigationWasAppInitiatedPromise);
    m_lastNavigationWasAppInitiatedPromise = WTFMove(promise);
    callOnMainThread([identifier = m_identifier, weakThis = WeakPtr { *this }]() mutable {
        if (auto* proxy = SWContextManager::singleton().serviceWorkerThreadProxy(identifier)) {
            proxy->thread().runLoop().postTaskForMode([weakThis = WTFMove(weakThis), appInitiated = proxy->lastNavigationWasAppInitiated()](auto&) {
                if (!weakThis || !weakThis->m_lastNavigationWasAppInitiatedPromise)
                    return;

                weakThis->m_lastNavigationWasAppInitiatedPromise->resolve<IDLBoolean>(appInitiated);
                weakThis->m_lastNavigationWasAppInitiatedPromise = nullptr;
            }, WorkerRunLoop::defaultMode());
        }
    });
}

RefPtr<PushSubscription> ServiceWorkerInternals::createPushSubscription(const String& endpoint, std::optional<EpochTimeStamp> expirationTime, const ArrayBuffer& serverVAPIDPublicKey, const ArrayBuffer& clientECDHPublicKey, const ArrayBuffer& auth)
{
    return PushSubscription::create(PushSubscriptionData { std::nullopt, { endpoint }, expirationTime, serverVAPIDPublicKey.toVector(), clientECDHPublicKey.toVector(), auth.toVector() });
}

bool ServiceWorkerInternals::fetchEventIsSameSite(FetchEvent& event)
{
    return event.request().internalRequest().isSameSite();
}

String ServiceWorkerInternals::serviceWorkerClientInternalIdentifier(const ServiceWorkerClient& client)
{
    return client.identifier().toString();
}

void ServiceWorkerInternals::setAsInspected(bool isInspected)
{
    SWContextManager::singleton().setAsInspected(m_identifier, isInspected);
}

void ServiceWorkerInternals::enableConsoleMessageReporting(ScriptExecutionContext& context)
{
    downcast<ServiceWorkerGlobalScope>(context).enableConsoleMessageReporting();
}

void ServiceWorkerInternals:: logReportedConsoleMessage(ScriptExecutionContext& context, const String& value)
{
    downcast<ServiceWorkerGlobalScope>(context).addConsoleMessage(MessageSource::Storage, MessageLevel::Info, value, 0);
}

} // namespace WebCore
