/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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

#include "SWServer.h"
#include "ServiceWorkerJobData.h"
#include "Timer.h"
#include "WorkerFetchResult.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class SWServerWorker;
class ServiceWorkerJob;
struct WorkerFetchResult;

class SWServerJobQueue final : public CanMakeCheckedPtr<SWServerJobQueue> {
    WTF_MAKE_TZONE_ALLOCATED(SWServerJobQueue);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SWServerJobQueue);
public:
    explicit SWServerJobQueue(SWServer&, const ServiceWorkerRegistrationKey&);
    SWServerJobQueue(const SWServerRegistration&) = delete;
    ~SWServerJobQueue();

    const ServiceWorkerJobData& firstJob() const { return m_jobQueue.first(); }
    const ServiceWorkerJobData& lastJob() const { return m_jobQueue.last(); }
    void enqueueJob(ServiceWorkerJobData&& jobData) { m_jobQueue.append(WTFMove(jobData)); }
    size_t size() const { return m_jobQueue.size(); }

    void runNextJob();

    void scriptFetchFinished(const ServiceWorkerJobDataIdentifier&, const std::optional<ProcessIdentifier>&, WorkerFetchResult&&);
    void importedScriptsFetchFinished(const ServiceWorkerJobDataIdentifier&, const Vector<std::pair<URL, ScriptBuffer>>&, const std::optional<ProcessIdentifier>&);
    void scriptContextFailedToStart(const ServiceWorkerJobDataIdentifier&, ServiceWorkerIdentifier, const String& message);
    void scriptContextStarted(const ServiceWorkerJobDataIdentifier&, ServiceWorkerIdentifier);
    void didFinishInstall(const ServiceWorkerJobDataIdentifier&, SWServerWorker&, bool wasSuccessful);
    void didResolveRegistrationPromise();
    void cancelJobsFromConnection(SWServerConnectionIdentifier);
    void cancelJobsFromServiceWorker(ServiceWorkerIdentifier);

    bool isCurrentlyProcessingJob(const ServiceWorkerJobDataIdentifier&) const;

private:
    void jobTimerFired();
    void runNextJobSynchronously();
    void rejectCurrentJob(const ExceptionData&);
    void finishCurrentJob();

    void runRegisterJob(const ServiceWorkerJobData&);
    void runUnregisterJob(const ServiceWorkerJobData&);
    void runUpdateJob(const ServiceWorkerJobData&);

    void install(SWServerRegistration&, ServiceWorkerIdentifier);

    void removeAllJobsMatching(const Function<bool(ServiceWorkerJobData&)>&);
    void scriptAndImportedScriptsFetchFinished(const ServiceWorkerJobData&, SWServerRegistration&);

    Ref<SWServer> protectedServer() const { return m_server.get(); }

    Deque<ServiceWorkerJobData> m_jobQueue;

    Timer m_jobTimer;
    WeakRef<SWServer> m_server;
    ServiceWorkerRegistrationKey m_registrationKey;
    WorkerFetchResult m_workerFetchResult;
};

} // namespace WebCore
