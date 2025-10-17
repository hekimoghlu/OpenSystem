/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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

#include "InspectorWebAgentBase.h"
#include "StorageArea.h"
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace Inspector {
class DOMStorageFrontendDispatcher;
}

namespace WebCore {

class LocalFrame;
class Page;
class SecurityOrigin;
class Storage;

class InspectorDOMStorageAgent final : public InspectorAgentBase, public Inspector::DOMStorageBackendDispatcherHandler {
    WTF_MAKE_NONCOPYABLE(InspectorDOMStorageAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorDOMStorageAgent);
public:
    InspectorDOMStorageAgent(PageAgentContext&);
    ~InspectorDOMStorageAgent();

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // DOMStorageBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::DOMStorage::Item>>> getDOMStorageItems(Ref<JSON::Object>&& storageId);
    Inspector::Protocol::ErrorStringOr<void> setDOMStorageItem(Ref<JSON::Object>&& storageId, const String& key, const String& value);
    Inspector::Protocol::ErrorStringOr<void> removeDOMStorageItem(Ref<JSON::Object>&& storageId, const String& key);
    Inspector::Protocol::ErrorStringOr<void> clearDOMStorageItems(Ref<JSON::Object>&& storageId);

    // InspectorInstrumentation
    void didDispatchDOMStorageEvent(const String& key, const String& oldValue, const String& newValue, StorageType, const SecurityOrigin&);

    // CommandLineAPI
    static String storageId(Storage&);
    static Ref<Inspector::Protocol::DOMStorage::StorageId> storageId(const SecurityOrigin&, bool isLocalStorage);

private:
    RefPtr<StorageArea> findStorageArea(Inspector::Protocol::ErrorString&, Ref<JSON::Object>&& storageId, LocalFrame*&);

    std::unique_ptr<Inspector::DOMStorageFrontendDispatcher> m_frontendDispatcher;
    RefPtr<Inspector::DOMStorageBackendDispatcher> m_backendDispatcher;

    WeakRef<Page> m_inspectedPage;
};

} // namespace WebCore
