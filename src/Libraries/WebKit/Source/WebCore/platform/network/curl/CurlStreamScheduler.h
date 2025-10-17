/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 18, 2024.
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

#include "CurlStream.h"
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class CurlStreamScheduler {
    WTF_MAKE_TZONE_ALLOCATED(CurlStreamScheduler);
    WTF_MAKE_NONCOPYABLE(CurlStreamScheduler);
public:
    CurlStreamScheduler();
    virtual ~CurlStreamScheduler();

    WEBCORE_EXPORT CurlStreamID createStream(const URL&, CurlStream::Client&, CurlStream::ServerTrustEvaluation, CurlStream::LocalhostAlias);
    WEBCORE_EXPORT void destroyStream(CurlStreamID);
    WEBCORE_EXPORT void send(CurlStreamID, UniqueArray<uint8_t>&&, size_t);

    void callOnWorkerThread(Function<void()>&&);
    void callClientOnMainThread(CurlStreamID, Function<void(CurlStream::Client&)>&&);

private:
    void startThreadIfNeeded();
    void stopThreadIfNoMoreJobRunning();

    void executeTasks();

    void workerThread();

    Lock m_mutex;
    RefPtr<Thread> m_thread;
    bool m_runThread { false };

    CurlStreamID m_currentStreamID = 1;

    Vector<Function<void()>> m_taskQueue;
    HashMap<CurlStreamID, CurlStream::Client*> m_clientList;
    HashMap<CurlStreamID, std::unique_ptr<CurlStream>> m_streamList;
};

} // namespace WebCore
