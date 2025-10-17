/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#include "WorkerBadgeProxy.h"
#include "WorkerGlobalScopeProxy.h"
#include "WorkerDebuggerProxy.h"
#include "WorkerLoaderProxy.h"
#include "WorkerObjectProxy.h"
#include <wtf/CheckedPtr.h>
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class DedicatedWorkerThread;
class WorkerInspectorProxy;
class WorkerUserGestureForwarder;

class WorkerMessagingProxy final : public ThreadSafeRefCounted<WorkerMessagingProxy>, public WorkerGlobalScopeProxy, public WorkerObjectProxy, public WorkerLoaderProxy, public WorkerDebuggerProxy, public WorkerBadgeProxy, public CanMakeThreadSafeCheckedPtr<WorkerMessagingProxy> {
    WTF_MAKE_TZONE_ALLOCATED(WorkerMessagingProxy);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WorkerMessagingProxy);
public:
    explicit WorkerMessagingProxy(Worker&);
    virtual ~WorkerMessagingProxy();

    uint32_t checkedPtrCount() const { return CanMakeThreadSafeCheckedPtr<WorkerMessagingProxy>::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const { return CanMakeThreadSafeCheckedPtr<WorkerMessagingProxy>::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const { CanMakeThreadSafeCheckedPtr<WorkerMessagingProxy>::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const { CanMakeThreadSafeCheckedPtr<WorkerMessagingProxy>::decrementCheckedPtrCount(); }

private:
    // Implementations of WorkerGlobalScopeProxy.
    // (Only use these functions in the worker object thread.)
    void startWorkerGlobalScope(const URL& scriptURL, PAL::SessionID, const String& name, WorkerInitializationData&&, const ScriptBuffer& sourceCode, const ContentSecurityPolicyResponseHeaders&, bool shouldBypassMainWorldContentSecurityPolicy, const CrossOriginEmbedderPolicy&, MonotonicTime timeOrigin, ReferrerPolicy, WorkerType, FetchRequestCredentials, JSC::RuntimeFlags) final;
    void terminateWorkerGlobalScope() final;
    void postMessageToWorkerGlobalScope(MessageWithMessagePorts&&) final;
    void postTaskToWorkerGlobalScope(Function<void(ScriptExecutionContext&)>&&) final;
    void workerObjectDestroyed() final;
    void notifyNetworkStateChange(bool isOnline) final;
    void suspendForBackForwardCache() final;
    void resumeForBackForwardCache() final;

    // Implementation of WorkerObjectProxy.
    // (Only use these functions in the worker context thread.)
    void postMessageToWorkerObject(MessageWithMessagePorts&&) final;
    void postTaskToWorkerObject(Function<void(Worker&)>&&) final;
    void postExceptionToWorkerObject(const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL) final;
    void reportErrorToWorkerObject(const String&) final;
    void workerGlobalScopeClosed() final;
    void workerGlobalScopeDestroyed() final;

    // Implementation of WorkerDebuggerProxy.
    // (Only use these functions in the worker context thread.)
    void postMessageToDebugger(const String&) final;
    void setResourceCachingDisabledByWebInspector(bool) final;

    // Implementation of WorkerLoaderProxy.
    // These functions are called on different threads to schedule loading
    // requests and to send callbacks back to WorkerGlobalScope.
    bool isWorkerMessagingProxy() const final { return true; }
    void postTaskToLoader(ScriptExecutionContext::Task&&) final;
    ScriptExecutionContextIdentifier loaderContextIdentifier() const final;
    RefPtr<CacheStorageConnection> createCacheStorageConnection() final;
    RefPtr<RTCDataChannelRemoteHandlerConnection> createRTCDataChannelRemoteHandlerConnection() final;

    void workerThreadCreated(DedicatedWorkerThread&);

    bool askedToTerminate() const final { return m_askedToTerminate; }

    void workerGlobalScopeDestroyedInternal();
    Worker* workerObject() const { return m_workerObject; }

    // WorkerBadgeProxy
    void setAppBadge(std::optional<uint64_t>) final;
    
    RefPtr<ScriptExecutionContext> m_scriptExecutionContext;
    ScriptExecutionContextIdentifier m_loaderContextIdentifier;
    RefPtr<WorkerInspectorProxy> m_inspectorProxy;
    RefPtr<WorkerUserGestureForwarder> m_userGestureForwarder;
    Worker* m_workerObject;
    bool m_mayBeDestroyed { false };
    RefPtr<DedicatedWorkerThread> m_workerThread;
    URL m_scriptURL;

    bool m_askedToSuspend { false };
    bool m_askedToTerminate { false };

    Vector<std::unique_ptr<ScriptExecutionContext::Task>> m_queuedEarlyTasks; // Tasks are queued here until there's a thread object created.
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::WorkerMessagingProxy)
    static bool isType(const WebCore::WorkerLoaderProxy& proxy) { return proxy.isWorkerMessagingProxy(); }
SPECIALIZE_TYPE_TRAITS_END()
