/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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
#pragma once

#include "EpochTimeStamp.h"
#include "IDLTypes.h"
#include "JSDOMPromiseDeferredForward.h"
#include "ServiceWorkerIdentifier.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {

class FetchEvent;
class FetchResponse;
class PushSubscription;
class ScriptExecutionContext;
class ServiceWorkerGlobalScope;
class ServiceWorkerClient;

template<typename IDLType> class DOMPromiseDeferred;

class WEBCORE_TESTSUPPORT_EXPORT ServiceWorkerInternals : public RefCountedAndCanMakeWeakPtr<ServiceWorkerInternals> {
public:
    static Ref<ServiceWorkerInternals> create(ServiceWorkerGlobalScope& globalScope, ServiceWorkerIdentifier identifier) { return adoptRef(*new ServiceWorkerInternals { globalScope, identifier }); }
    ~ServiceWorkerInternals();

    void setOnline(bool isOnline);
    void terminate();

    void waitForFetchEventToFinish(FetchEvent&, DOMPromiseDeferred<IDLInterface<FetchResponse>>&&);
    Ref<FetchEvent> createBeingDispatchedFetchEvent(ScriptExecutionContext&);
    Ref<FetchResponse> createOpaqueWithBlobBodyResponse(ScriptExecutionContext&);

    void schedulePushEvent(const String&, RefPtr<DeferredPromise>&&);
    void schedulePushSubscriptionChangeEvent(PushSubscription* newSubscription, PushSubscription* oldSubscription);
    Vector<String> fetchResponseHeaderList(FetchResponse&);

    String processName() const;

    bool isThrottleable() const;

    int processIdentifier() const;

    void lastNavigationWasAppInitiated(Ref<DeferredPromise>&&);
    
    RefPtr<PushSubscription> createPushSubscription(const String& endpoint, std::optional<EpochTimeStamp> expirationTime, const ArrayBuffer& serverVAPIDPublicKey, const ArrayBuffer& clientECDHPublicKey, const ArrayBuffer& auth);

    bool fetchEventIsSameSite(FetchEvent&);

    String serviceWorkerClientInternalIdentifier(const ServiceWorkerClient&);
    void setAsInspected(bool);
    void enableConsoleMessageReporting(ScriptExecutionContext&);
    void logReportedConsoleMessage(ScriptExecutionContext&, const String&);

private:
    ServiceWorkerInternals(ServiceWorkerGlobalScope&, ServiceWorkerIdentifier);

    ServiceWorkerIdentifier m_identifier;
    RefPtr<DeferredPromise> m_lastNavigationWasAppInitiatedPromise;
    HashMap<uint64_t, RefPtr<DeferredPromise>> m_pushEventPromises;
    uint64_t m_pushEventCounter { 0 };
};

} // namespace WebCore
