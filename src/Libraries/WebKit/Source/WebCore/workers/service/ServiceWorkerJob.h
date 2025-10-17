/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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

#include "ResourceLoaderIdentifier.h"
#include "ResourceResponse.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerJobClient.h"
#include "ServiceWorkerJobData.h"
#include "ServiceWorkerTypes.h"
#include "WorkerScriptLoader.h"
#include "WorkerScriptLoaderClient.h"
#include <wtf/CompletionHandler.h>
#include <wtf/RefPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Threading.h>

namespace WebCore {

class DeferredPromise;
class Exception;
class ScriptExecutionContext;
enum class ServiceWorkerJobType : uint8_t;
struct ServiceWorkerRegistrationData;

class ServiceWorkerJob : public WorkerScriptLoaderClient {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ServiceWorkerJob, WEBCORE_EXPORT);
public:
    ServiceWorkerJob(ServiceWorkerJobClient&, RefPtr<DeferredPromise>&&, ServiceWorkerJobData&&);
    WEBCORE_EXPORT ~ServiceWorkerJob();

    void failedWithException(const Exception&);
    void resolvedWithRegistration(ServiceWorkerRegistrationData&&, ShouldNotifyWhenResolved);
    void resolvedWithUnregistrationResult(bool);
    void startScriptFetch(FetchOptions::Cache);

    using Identifier = ServiceWorkerJobIdentifier;
    Identifier identifier() const { return m_jobData.identifier().jobIdentifier; }

    const ServiceWorkerJobData& data() const { return m_jobData; }
    bool hasPromise() const { return !!m_promise; }
    RefPtr<DeferredPromise> takePromise();

    void fetchScriptWithContext(ScriptExecutionContext&, FetchOptions::Cache);

    const ServiceWorkerOrClientIdentifier& contextIdentifier() { return m_contextIdentifier; }

    bool cancelPendingLoad();

    WEBCORE_EXPORT static ResourceError validateServiceWorkerResponse(const ServiceWorkerJobData&, const ResourceResponse&);

private:
    // WorkerScriptLoaderClient
    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) final;
    void notifyFinished(std::optional<ScriptExecutionContextIdentifier>) final;

    ServiceWorkerJobClient& m_client;
    ServiceWorkerJobData m_jobData;
    RefPtr<DeferredPromise> m_promise;

    bool m_completed { false };

    ServiceWorkerOrClientIdentifier m_contextIdentifier;
    RefPtr<WorkerScriptLoader> m_scriptLoader;

#if ASSERT_ENABLED
    Ref<Thread> m_creationThread { Thread::current() };
#endif
};

} // namespace WebCore
