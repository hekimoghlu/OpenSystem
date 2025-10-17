/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 9, 2023.
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

#include "CurlContext.h"
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Threading.h>

namespace WebCore {

class CurlRequestSchedulerClient;

class CurlRequestScheduler {
    WTF_MAKE_TZONE_ALLOCATED(CurlRequestScheduler);
    WTF_MAKE_NONCOPYABLE(CurlRequestScheduler);
    friend NeverDestroyed<CurlRequestScheduler>;
public:
    CurlRequestScheduler(long maxConnects, long maxTotalConnections, long maxHostConnections);
    ~CurlRequestScheduler() { stopThread(); }

    bool add(CurlRequestSchedulerClient*);
    void cancel(CurlRequestSchedulerClient*);

    void callOnWorkerThread(Function<void()>&&);

private:
    void startOrWakeUpThread();
    void wakeUpThreadIfPossible();
    void stopThreadIfNoMoreJobRunning();
    void stopThread();

    void executeTasks();

    void workerThread();

    void startTransfer(CurlRequestSchedulerClient*);
    void completeTransfer(CurlRequestSchedulerClient*, CURLcode);
    void cancelTransfer(CurlRequestSchedulerClient*);
    void finalizeTransfer(CurlRequestSchedulerClient*, Function<void()>);

    Lock m_mutex;
    RefPtr<Thread> m_thread;
    bool m_runThread { false };

    Vector<Function<void()>> m_taskQueue;
    UncheckedKeyHashSet<CurlRequestSchedulerClient*> m_activeJobs;
    HashMap<CURL*, CurlRequestSchedulerClient*> m_clientMaps;

    Lock m_multiHandleMutex;
    std::optional<CurlMultiHandle> m_curlMultiHandle;

    long m_maxConnects;
    long m_maxTotalConnections;
    long m_maxHostConnections;
};

} // namespace WebCore
