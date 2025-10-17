/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace Inspector {
class InjectedScriptManager;
}

namespace WebCore {

class Page;

class InspectorIndexedDBAgent final : public InspectorAgentBase, public Inspector::IndexedDBBackendDispatcherHandler {
    WTF_MAKE_TZONE_ALLOCATED(InspectorIndexedDBAgent);
    WTF_MAKE_NONCOPYABLE(InspectorIndexedDBAgent);
public:
    InspectorIndexedDBAgent(PageAgentContext&);
    ~InspectorIndexedDBAgent();

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // IndexedDBBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    void requestDatabaseNames(const String& securityOrigin, Ref<RequestDatabaseNamesCallback>&&);
    void requestDatabase(const String& securityOrigin, const String& databaseName, Ref<RequestDatabaseCallback>&&);
    void requestData(const String& securityOrigin, const String& databaseName, const String& objectStoreName, const String& indexName, int skipCount, int pageSize, RefPtr<JSON::Object>&& keyRange, Ref<RequestDataCallback>&&);
    void clearObjectStore(const String& securityOrigin, const String& databaseName, const String& objectStoreName, Ref<ClearObjectStoreCallback>&&);

private:
    Inspector::InjectedScriptManager& m_injectedScriptManager;
    RefPtr<Inspector::IndexedDBBackendDispatcher> m_backendDispatcher;

    WeakRef<Page> m_inspectedPage;
};

} // namespace WebCore
