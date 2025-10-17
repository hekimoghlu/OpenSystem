/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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

#include "HeapObserver.h"
#include "InspectorAgentBase.h"
#include "InspectorBackendDispatchers.h"
#include "InspectorFrontendDispatchers.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
struct HeapSnapshotNode;
}

namespace Inspector {

class InjectedScriptManager;

class JS_EXPORT_PRIVATE InspectorHeapAgent : public InspectorAgentBase, public HeapBackendDispatcherHandler, public JSC::HeapObserver {
    WTF_MAKE_NONCOPYABLE(InspectorHeapAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorHeapAgent);
public:
    InspectorHeapAgent(AgentContext&);
    ~InspectorHeapAgent() override;

    // InspectorAgentBase
    void didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*) final;
    void willDestroyFrontendAndBackend(DisconnectReason) final;

    // HeapBackendDispatcherHandler
    Protocol::ErrorStringOr<void> enable() override;
    Protocol::ErrorStringOr<void> disable() override;
    Protocol::ErrorStringOr<void> gc() final;
    Protocol::ErrorStringOr<std::tuple<double, Protocol::Heap::HeapSnapshotData>> snapshot() final;
    Protocol::ErrorStringOr<void> startTracking() final;
    Protocol::ErrorStringOr<void> stopTracking() final;
    Protocol::ErrorStringOr<std::tuple<String, RefPtr<Protocol::Debugger::FunctionDetails>, RefPtr<Protocol::Runtime::ObjectPreview>>> getPreview(int heapObjectId) final;
    Protocol::ErrorStringOr<Ref<Protocol::Runtime::RemoteObject>> getRemoteObject(int heapObjectId, const String& objectGroup) final;

    // JSC::HeapObserver
    void willGarbageCollect() final;
    void didGarbageCollect(JSC::CollectionScope) final;

protected:
    void clearHeapSnapshots();

    virtual void dispatchGarbageCollectedEvent(Protocol::Heap::GarbageCollection::Type, Seconds startTime, Seconds endTime);

private:
    std::optional<JSC::HeapSnapshotNode> nodeForHeapObjectIdentifier(Protocol::ErrorString&, unsigned heapObjectIdentifier);

    InjectedScriptManager& m_injectedScriptManager;
    std::unique_ptr<HeapFrontendDispatcher> m_frontendDispatcher;
    RefPtr<HeapBackendDispatcher> m_backendDispatcher;
    InspectorEnvironment& m_environment;

    bool m_enabled { false };
    bool m_tracking { false };
    Seconds m_gcStartTime { Seconds::nan() };
};

} // namespace Inspector
