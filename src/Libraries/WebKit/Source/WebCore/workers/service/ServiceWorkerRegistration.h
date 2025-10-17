/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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

#include "ActiveDOMObject.h"
#include "CookieStoreManager.h"
#include "EventTarget.h"
#include "JSDOMPromiseDeferredForward.h"
#include "Notification.h"
#include "NotificationOptions.h"
#include "PushPermissionState.h"
#include "PushSubscription.h"
#include "PushSubscriptionOwner.h"
#include "SWClientConnection.h"
#include "ServiceWorkerRegistrationData.h"
#include "Supplementable.h"
#include "Timer.h"
#include <wtf/Forward.h>
#include <wtf/ListHashSet.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

class DeferredPromise;
class NavigationPreloadManager;
class ScriptExecutionContext;
class ServiceWorker;
class ServiceWorkerContainer;
class WebCoreOpaqueRoot;
struct CookieStoreGetOptions;

class ServiceWorkerRegistration final : public RefCounted<ServiceWorkerRegistration>, public Supplementable<ServiceWorkerRegistration>, public EventTarget, public ActiveDOMObject, public PushSubscriptionOwner {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(ServiceWorkerRegistration, WEBCORE_EXPORT);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<ServiceWorkerRegistration> getOrCreate(ScriptExecutionContext&, Ref<ServiceWorkerContainer>&&, ServiceWorkerRegistrationData&&);

    WEBCORE_EXPORT ~ServiceWorkerRegistration();

    ServiceWorkerRegistrationIdentifier identifier() const { return m_registrationData.identifier; }

    ServiceWorker* installing();
    ServiceWorker* waiting();
    ServiceWorker* active();

    bool isActive() const final { return !!m_activeWorker; }

    ServiceWorker* getNewestWorker() const;

    const String& scope() const;

    ServiceWorkerUpdateViaCache updateViaCache() const;
    void setUpdateViaCache(ServiceWorkerUpdateViaCache);

    WallTime lastUpdateTime() const;
    void setLastUpdateTime(WallTime);

    bool needsUpdate() const { return lastUpdateTime() && (WallTime::now() - lastUpdateTime()) > 86400_s; }

    void update(Ref<DeferredPromise>&&);
    void unregister(Ref<DeferredPromise>&&);

    void subscribeToPushService(const Vector<uint8_t>& applicationServerKey, DOMPromiseDeferred<IDLInterface<PushSubscription>>&&);
    void unsubscribeFromPushService(std::optional<PushSubscriptionIdentifier>, DOMPromiseDeferred<IDLBoolean>&&);
    void getPushSubscription(DOMPromiseDeferred<IDLNullable<IDLInterface<PushSubscription>>>&&);
    void getPushPermissionState(DOMPromiseDeferred<IDLEnumeration<PushPermissionState>>&&);

    const ServiceWorkerRegistrationData& data() const { return m_registrationData; }

    void updateStateFromServer(ServiceWorkerRegistrationState, RefPtr<ServiceWorker>&&);
    void queueTaskToFireUpdateFoundEvent();

    NavigationPreloadManager& navigationPreload();
    ServiceWorkerContainer& container() { return m_container.get(); }
    Ref<ServiceWorkerContainer> protectedContainer() const;

#if ENABLE(NOTIFICATION_EVENT)
    struct GetNotificationOptions {
        String tag;
    };

    void showNotification(ScriptExecutionContext&, String&& title, NotificationOptions&&, Ref<DeferredPromise>&&);
    void getNotifications(const GetNotificationOptions& filter, DOMPromiseDeferred<IDLSequence<IDLInterface<Notification>>>);
#endif

    CookieStoreManager& cookies();
    void addCookieChangeSubscriptions(Vector<CookieStoreGetOptions>&&, Ref<DeferredPromise>&&);
    void removeCookieChangeSubscriptions(Vector<CookieStoreGetOptions>&&, Ref<DeferredPromise>&&);
    void cookieChangeSubscriptions(Ref<DeferredPromise>&&);

private:
    ServiceWorkerRegistration(ScriptExecutionContext&, Ref<ServiceWorkerContainer>&&, ServiceWorkerRegistrationData&&);

    enum EventTargetInterfaceType eventTargetInterface() const final;
    ScriptExecutionContext* scriptExecutionContext() const final;
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    // ActiveDOMObject.
    void stop() final;
    bool virtualHasPendingActivity() const final;

    ServiceWorkerRegistrationData m_registrationData;
    Ref<ServiceWorkerContainer> m_container;

    RefPtr<ServiceWorker> m_installingWorker;
    RefPtr<ServiceWorker> m_waitingWorker;
    RefPtr<ServiceWorker> m_activeWorker;

    std::unique_ptr<NavigationPreloadManager> m_navigationPreload;

    RefPtr<CookieStoreManager> m_cookieStoreManager;
};

WebCoreOpaqueRoot root(ServiceWorkerRegistration*);

} // namespace WebCore
