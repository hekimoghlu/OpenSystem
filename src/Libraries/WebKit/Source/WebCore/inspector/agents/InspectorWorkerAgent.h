/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#include "WorkerInspectorProxy.h"
#include <JavaScriptCore/InspectorBackendDispatchers.h>
#include <JavaScriptCore/InspectorFrontendDispatchers.h>
#include <wtf/CheckedPtr.h>
#include <wtf/CheckedRef.h>
#include <wtf/FastMalloc.h>
#include <wtf/Lock.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class InspectorWorkerAgent : public InspectorAgentBase, public Inspector::WorkerBackendDispatcherHandler, public CanMakeThreadSafeCheckedPtr<InspectorWorkerAgent> {
    WTF_MAKE_NONCOPYABLE(InspectorWorkerAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorWorkerAgent);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(InspectorWorkerAgent);

public:
    ~InspectorWorkerAgent();

    Inspector::WorkerFrontendDispatcher& frontendDispatcher() { return *m_frontendDispatcher; }

    // InspectorAgentBase
    void didCreateFrontendAndBackend(Inspector::FrontendRouter*, Inspector::BackendDispatcher*);
    void willDestroyFrontendAndBackend(Inspector::DisconnectReason);

    // WorkerBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<void> enable();
    Inspector::Protocol::ErrorStringOr<void> disable();
    Inspector::Protocol::ErrorStringOr<void> initialized(const String& workerId);
    Inspector::Protocol::ErrorStringOr<void> sendMessageToWorker(const String& workerId, const String& message);

    // InspectorInstrumentation
    bool shouldWaitForDebuggerOnStart() const;
    void workerStarted(WorkerInspectorProxy&);
    void workerTerminated(WorkerInspectorProxy&);

protected:
    InspectorWorkerAgent(WebAgentContext&);

    virtual void connectToAllWorkerInspectorProxies() = 0;

    void connectToWorkerInspectorProxy(WorkerInspectorProxy&);

private:
    class PageChannel final : public WorkerInspectorProxy::PageChannel, public ThreadSafeRefCounted<PageChannel> {
        WTF_MAKE_TZONE_ALLOCATED_INLINE(PageChannel);
        WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PageChannel);

    public:
        static Ref<PageChannel> create(InspectorWorkerAgent&);

        void ref() const final { ThreadSafeRefCounted::ref(); }
        void deref() const final { ThreadSafeRefCounted::deref(); }

        void detachFromParentAgent();
        void sendMessageFromWorkerToFrontend(WorkerInspectorProxy&, String&&);

    private:
        explicit PageChannel(InspectorWorkerAgent&);

        Lock m_parentAgentLock;
        CheckedPtr<InspectorWorkerAgent> m_parentAgent WTF_GUARDED_BY_LOCK(m_parentAgentLock);
    };

    void disconnectFromAllWorkerInspectorProxies();
    void disconnectFromWorkerInspectorProxy(WorkerInspectorProxy&);

    const Ref<PageChannel> m_pageChannel;

    UniqueRef<Inspector::WorkerFrontendDispatcher> m_frontendDispatcher;
    RefPtr<Inspector::WorkerBackendDispatcher> m_backendDispatcher;

    MemoryCompactRobinHoodHashMap<String, WeakPtr<WorkerInspectorProxy>> m_connectedProxies;
    bool m_enabled { false };
};

} // namespace WebCore
