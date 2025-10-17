/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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

#include <WebCore/SWRegistrationStore.h>
#include <WebCore/ServiceWorkerContextData.h>
#include <WebCore/Timer.h>
#include <wtf/CompletionHandler.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {
class WebSWRegistrationStore;
}

namespace WebCore {
class SWServer;
}

namespace WebKit {

class NetworkStorageManager;

class WebSWRegistrationStore final : public WebCore::SWRegistrationStore {
    WTF_MAKE_TZONE_ALLOCATED(WebSWRegistrationStore);
public:
    static Ref<WebSWRegistrationStore> create(WebCore::SWServer&, NetworkStorageManager&);

private:
    WebSWRegistrationStore(WebCore::SWServer&, NetworkStorageManager&);

    // WebCore::SWRegistrationStore
    void clearAll(CompletionHandler<void()>&&);
    void flushChanges(CompletionHandler<void()>&&);
    void closeFiles(CompletionHandler<void()>&&);
    void importRegistrations(CompletionHandler<void(std::optional<Vector<WebCore::ServiceWorkerContextData>>)>&&);
    void updateRegistration(const WebCore::ServiceWorkerContextData&);
    void removeRegistration(const WebCore::ServiceWorkerRegistrationKey&);

    void scheduleUpdateIfNecessary();
    void updateToStorage(CompletionHandler<void()>&&);
    void updateTimerFired() { updateToStorage([] { }); }

    CheckedPtr<NetworkStorageManager> checkedManager() const;
    RefPtr<WebCore::SWServer> protectedServer() const;

    WeakPtr<WebCore::SWServer> m_server;
    WeakPtr<NetworkStorageManager> m_manager;
    WebCore::Timer m_updateTimer;
    HashMap<WebCore::ServiceWorkerRegistrationKey, std::optional<WebCore::ServiceWorkerContextData>> m_updates;
};

} // namespace WebKit
