/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 11, 2022.
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
#include "WebSWRegistrationStore.h"

#include "NetworkStorageManager.h"
#include <WebCore/SWServer.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSWRegistrationStore);

Ref<WebSWRegistrationStore> WebSWRegistrationStore::create(WebCore::SWServer& server, NetworkStorageManager& manager)
{
    return adoptRef(*new WebSWRegistrationStore(server, manager));
}

WebSWRegistrationStore::WebSWRegistrationStore(WebCore::SWServer& server, NetworkStorageManager& manager)
    : m_server(server)
    , m_manager(manager)
    , m_updateTimer(*this, &WebSWRegistrationStore::updateTimerFired)
{
    ASSERT(RunLoop::isMain());
}

CheckedPtr<NetworkStorageManager> WebSWRegistrationStore::checkedManager() const
{
    return m_manager.get();
}

RefPtr<WebCore::SWServer> WebSWRegistrationStore::protectedServer() const
{
    return m_server.get();
}

void WebSWRegistrationStore::clearAll(CompletionHandler<void()>&& callback)
{
    m_updates.clear();
    m_updateTimer.stop();
    if (!m_manager)
        return callback();

    checkedManager()->clearServiceWorkerRegistrations(WTFMove(callback));
}

void WebSWRegistrationStore::flushChanges(CompletionHandler<void()>&& callback)
{
    if (m_updateTimer.isActive())
        m_updateTimer.stop();

    updateToStorage(WTFMove(callback));
}

void WebSWRegistrationStore::closeFiles(CompletionHandler<void()>&& callback)
{
    if (!m_manager)
        return callback();

    checkedManager()->closeServiceWorkerRegistrationFiles(WTFMove(callback));
}

void WebSWRegistrationStore::importRegistrations(CompletionHandler<void(std::optional<Vector<WebCore::ServiceWorkerContextData>>)>&& callback)
{
    if (!m_manager)
        return callback(std::nullopt);

    checkedManager()->importServiceWorkerRegistrations(WTFMove(callback));
}

void WebSWRegistrationStore::updateRegistration(const WebCore::ServiceWorkerContextData& registration)
{
    m_updates.set(registration.registration.key, registration);
    scheduleUpdateIfNecessary();
}

void WebSWRegistrationStore::removeRegistration(const WebCore::ServiceWorkerRegistrationKey& key)
{
    m_updates.set(key, std::nullopt);
    scheduleUpdateIfNecessary();
}

void WebSWRegistrationStore::scheduleUpdateIfNecessary()
{
    ASSERT(RunLoop::isMain());

    if (m_updateTimer.isActive())
        return;

    m_updateTimer.startOneShot(0_s);
}

void WebSWRegistrationStore::updateToStorage(CompletionHandler<void()>&& callback)
{
    ASSERT(RunLoop::isMain());

    Vector<WebCore::ServiceWorkerRegistrationKey> registrationsToDelete;
    Vector<WebCore::ServiceWorkerContextData> registrationsToUpdate;
    for (auto& [key, registation] : m_updates) {
        if (!registation)
            registrationsToDelete.append(key);
        else
            registrationsToUpdate.append(WTFMove(*registation));
    }
    m_updates.clear();

    if (!m_manager)
        return callback();

    checkedManager()->updateServiceWorkerRegistrations(WTFMove(registrationsToUpdate), WTFMove(registrationsToDelete), [this, weakThis = WeakPtr { *this }, callback = WTFMove(callback)](auto&& result) mutable {
        ASSERT(RunLoop::isMain());

        if (!weakThis || !m_server || !result)
            return callback();

        auto allScripts = WTFMove(result.value());
        for (auto&& scripts : allScripts)
            protectedServer()->didSaveWorkerScriptsToDisk(scripts.identifier, WTFMove(scripts.mainScript), WTFMove(scripts.importedScripts));

        callback();
    });
}

} // namespace WebKit
