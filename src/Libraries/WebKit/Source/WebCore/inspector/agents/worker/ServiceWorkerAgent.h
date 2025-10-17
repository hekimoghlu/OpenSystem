/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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

namespace WebCore {

class ServiceWorkerGlobalScope;

class ServiceWorkerAgent final : public InspectorAgentBase, public Inspector::ServiceWorkerBackendDispatcherHandler {
    WTF_MAKE_NONCOPYABLE(ServiceWorkerAgent);
    WTF_MAKE_TZONE_ALLOCATED(ServiceWorkerAgent);
public:
    ServiceWorkerAgent(WorkerAgentContext&);
    ~ServiceWorkerAgent();

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // ServiceWorkerBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<Ref<Inspector::Protocol::ServiceWorker::Configuration>> getInitializationInfo();

private:
    WeakRef<ServiceWorkerGlobalScope, WeakPtrImplWithEventTargetData> m_serviceWorkerGlobalScope;
    RefPtr<Inspector::ServiceWorkerBackendDispatcher> m_backendDispatcher;
};

} // namespace WebCore
