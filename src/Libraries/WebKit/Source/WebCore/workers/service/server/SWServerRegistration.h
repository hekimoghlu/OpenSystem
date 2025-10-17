/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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

#include "CookieChangeSubscription.h"
#include "NavigationPreloadState.h"
#include "SWServer.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerRegistrationData.h"
#include "ServiceWorkerTypes.h"
#include "Timer.h"
#include <wtf/Forward.h>
#include <wtf/HashCountedSet.h>
#include <wtf/HashSet.h>
#include <wtf/Identified.h>
#include <wtf/MonotonicTime.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WallTime.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class SWServer;
class SWServerWorker;
enum class ServiceWorkerRegistrationState : uint8_t;
enum class ServiceWorkerState : uint8_t;
struct ExceptionData;
struct ServiceWorkerContextData;

enum class IsAppInitiated : bool { No, Yes };

class SWServerRegistration : public RefCountedAndCanMakeWeakPtr<SWServerRegistration>, public Identified<ServiceWorkerRegistrationIdentifier> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(SWServerRegistration, WEBCORE_EXPORT);
public:
    static Ref<SWServerRegistration> create(SWServer&, const ServiceWorkerRegistrationKey&, ServiceWorkerUpdateViaCache, const URL& scopeURL, const URL& scriptURL, std::optional<ScriptExecutionContextIdentifier> serviceWorkerPageIdentifier, NavigationPreloadState&&);
    WEBCORE_EXPORT ~SWServerRegistration();

    const ServiceWorkerRegistrationKey& key() const { return m_registrationKey; }

    SWServerWorker* getNewestWorker();
    WEBCORE_EXPORT ServiceWorkerRegistrationData data() const;

    void setLastUpdateTime(WallTime);
    WallTime lastUpdateTime() const { return m_lastUpdateTime; }
    bool isStale() const { return m_lastUpdateTime && (WallTime::now() - m_lastUpdateTime) > 86400_s; }

    void setUpdateViaCache(ServiceWorkerUpdateViaCache);
    ServiceWorkerUpdateViaCache updateViaCache() const { return m_updateViaCache; }

    void updateRegistrationState(ServiceWorkerRegistrationState, SWServerWorker*);
    void updateWorkerState(SWServerWorker&, ServiceWorkerState);
    void fireUpdateFoundEvent();

    void addClientServiceWorkerRegistration(SWServerConnectionIdentifier);
    void removeClientServiceWorkerRegistration(SWServerConnectionIdentifier);

    void setPreInstallationWorker(SWServerWorker*);
    SWServerWorker* preInstallationWorker() const { return m_preInstallationWorker.get(); }
    SWServerWorker* installingWorker() const { return m_installingWorker.get(); }
    SWServerWorker* waitingWorker() const { return m_waitingWorker.get(); }
    SWServerWorker* activeWorker() const { return m_activeWorker.get(); }

    MonotonicTime creationTime() const { return m_creationTime; }

    bool hasClientsUsingRegistration() const { return !m_clientsUsingRegistration.isEmpty(); }
    void addClientUsingRegistration(const ScriptExecutionContextIdentifier&);
    void removeClientUsingRegistration(const ScriptExecutionContextIdentifier&);
    void unregisterServerConnection(SWServerConnectionIdentifier);

    void notifyClientsOfControllerChange();
    void controlClient(ScriptExecutionContextIdentifier);

    void clear();
    bool tryClear();
    void tryActivate();
    void didFinishActivation(ServiceWorkerIdentifier);
    
    bool isUnregistered() const;

    void forEachConnection(const Function<void(SWServer::Connection&)>&);

    WEBCORE_EXPORT bool shouldSoftUpdate(const FetchOptions&) const;
    WEBCORE_EXPORT void scheduleSoftUpdate(IsAppInitiated);
    static constexpr Seconds softUpdateDelay { 1_s };

    URL scopeURLWithoutFragment() const { return m_scopeURL; }
    URL scriptURL() const { return m_scriptURL; }

    bool isAppInitiated() const { return m_isAppInitiated; }
    std::optional<ScriptExecutionContextIdentifier> serviceWorkerPageIdentifier() const { return m_serviceWorkerPageIdentifier; }

    WEBCORE_EXPORT std::optional<ExceptionData> enableNavigationPreload();
    WEBCORE_EXPORT std::optional<ExceptionData> disableNavigationPreload();
    WEBCORE_EXPORT std::optional<ExceptionData> setNavigationPreloadHeaderValue(String&&);
    const NavigationPreloadState& navigationPreloadState() const { return m_preloadState; }

    WEBCORE_EXPORT void addCookieChangeSubscriptions(Vector<CookieChangeSubscription>&&);
    WEBCORE_EXPORT void removeCookieChangeSubscriptions(Vector<CookieChangeSubscription>&&);
    WEBCORE_EXPORT Vector<CookieChangeSubscription> cookieChangeSubscriptions() const;

private:
    SWServerRegistration(SWServer&, const ServiceWorkerRegistrationKey&, ServiceWorkerUpdateViaCache, const URL& scopeURL, const URL& scriptURL, std::optional<ScriptExecutionContextIdentifier> serviceWorkerPageIdentifier, NavigationPreloadState&&);

    void activate();
    void handleClientUnload();
    void softUpdate();

    RefPtr<SWServer> protectedServer() const { return m_server.get(); }

    ServiceWorkerRegistrationKey m_registrationKey;
    ServiceWorkerUpdateViaCache m_updateViaCache;
    URL m_scopeURL;
    URL m_scriptURL;
    std::optional<ScriptExecutionContextIdentifier> m_serviceWorkerPageIdentifier;

    RefPtr<SWServerWorker> m_preInstallationWorker; // Implementation detail, not part of the specification.
    RefPtr<SWServerWorker> m_installingWorker;
    RefPtr<SWServerWorker> m_waitingWorker;
    RefPtr<SWServerWorker> m_activeWorker;

    HashSet<CookieChangeSubscription> m_cookieChangeSubscriptions;

    WallTime m_lastUpdateTime;
    
    HashCountedSet<SWServerConnectionIdentifier> m_connectionsWithClientRegistrations;
    WeakPtr<SWServer> m_server;

    MonotonicTime m_creationTime;
    HashMap<SWServerConnectionIdentifier, HashSet<ScriptExecutionContextIdentifier>> m_clientsUsingRegistration;

    WebCore::Timer m_softUpdateTimer;
    
    bool m_isAppInitiated { true };
    NavigationPreloadState m_preloadState;
};

} // namespace WebCore
