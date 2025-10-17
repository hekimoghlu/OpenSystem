/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 30, 2022.
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

#include "ClientOrigin.h"
#include "SharedWorkerIdentifier.h"
#include "WorkerBadgeProxy.h"
#include "WorkerDebuggerProxy.h"
#include "WorkerLoaderProxy.h"
#include "WorkerObjectProxy.h"
#include "WorkerOptions.h"
#include <wtf/CheckedPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class CacheStorageProvider;
class Page;
class SharedWorker;
class SharedWorkerThread;

struct WorkerFetchResult;
struct WorkerInitializationData;

class SharedWorkerThreadProxy final : public ThreadSafeRefCounted<SharedWorkerThreadProxy>, public WorkerObjectProxy, public WorkerLoaderProxy, public WorkerDebuggerProxy, public WorkerBadgeProxy, public CanMakeWeakPtr<SharedWorkerThreadProxy>, public CanMakeThreadSafeCheckedPtr<SharedWorkerThreadProxy> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SharedWorkerThreadProxy);
public:
    template<typename... Args> static Ref<SharedWorkerThreadProxy> create(Args&&... args) { return adoptRef(*new SharedWorkerThreadProxy(std::forward<Args>(args)...)); }
    WEBCORE_EXPORT ~SharedWorkerThreadProxy();

    static SharedWorkerThreadProxy* byIdentifier(ScriptExecutionContextIdentifier);
    WEBCORE_EXPORT static bool hasInstances();

    SharedWorkerIdentifier identifier() const;
    SharedWorkerThread& thread() { return m_workerThread; }

    bool isTerminatingOrTerminated() const { return m_isTerminatingOrTerminated; }
    void setAsTerminatingOrTerminated() { m_isTerminatingOrTerminated = true; }

    uint32_t checkedPtrCount() const { return CanMakeThreadSafeCheckedPtr<SharedWorkerThreadProxy>::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const { return CanMakeThreadSafeCheckedPtr<SharedWorkerThreadProxy>::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const { CanMakeThreadSafeCheckedPtr<SharedWorkerThreadProxy>::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const { CanMakeThreadSafeCheckedPtr<SharedWorkerThreadProxy>::decrementCheckedPtrCount(); }

private:
    WEBCORE_EXPORT SharedWorkerThreadProxy(Ref<Page>&&, SharedWorkerIdentifier, const ClientOrigin&, WorkerFetchResult&&, WorkerOptions&&, WorkerInitializationData&&, CacheStorageProvider&);

    bool postTaskForModeToWorkerOrWorkletGlobalScope(ScriptExecutionContext::Task&&, const String& mode);

    // WorkerObjectProxy.
    void postExceptionToWorkerObject(const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL) final;
    void reportErrorToWorkerObject(const String&) final;
    void postMessageToWorkerObject(MessageWithMessagePorts&&) final { }
    void workerGlobalScopeDestroyed() final { }
    void workerGlobalScopeClosed() final;

    // WorkerLoaderProxy.
    RefPtr<CacheStorageConnection> createCacheStorageConnection() final;
    RefPtr<RTCDataChannelRemoteHandlerConnection> createRTCDataChannelRemoteHandlerConnection() final;
    void postTaskToLoader(ScriptExecutionContext::Task&&) final;
    ScriptExecutionContextIdentifier loaderContextIdentifier() const final;

    // WorkerDebuggerProxy.
    void postMessageToDebugger(const String&) final;
    void setResourceCachingDisabledByWebInspector(bool) final;

    // WorkerBadgeProxy
    void setAppBadge(std::optional<uint64_t>) final;

    static void networkStateChanged(bool isOnLine);
    void notifyNetworkStateChange(bool isOnline);

    ReportingClient* reportingClient() const final;

    Ref<Page> m_page;
    Ref<Document> m_document;
    ScriptExecutionContextIdentifier m_contextIdentifier;
    Ref<SharedWorkerThread> m_workerThread;
    CacheStorageProvider& m_cacheStorageProvider;
    RefPtr<CacheStorageConnection> m_cacheStorageConnection;
    bool m_isTerminatingOrTerminated { false };
    ClientOrigin m_clientOrigin;
};

} // namespace WebCore
