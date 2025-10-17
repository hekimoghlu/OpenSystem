/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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

#include "Page.h"
#include "ResourceLoaderIdentifier.h"
#include "Timer.h"
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class LocalFrame;
class ResourceResponse;
class ProgressTrackerClient;
struct ProgressItem;

class ProgressTracker final : public CanMakeCheckedPtr<ProgressTracker> {
    WTF_MAKE_NONCOPYABLE(ProgressTracker);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ProgressTracker);
public:
    explicit ProgressTracker(Page&, UniqueRef<ProgressTrackerClient>&&);
    ~ProgressTracker();

    ProgressTrackerClient& client() { return m_client.get(); }

    double estimatedProgress() const { return m_progressValue; }

    void progressStarted(LocalFrame&);
    void progressCompleted(LocalFrame&);

    void incrementProgress(ResourceLoaderIdentifier, const ResourceResponse&);
    void incrementProgress(ResourceLoaderIdentifier, unsigned bytesReceived);
    void completeProgress(ResourceLoaderIdentifier);

    long long totalPageAndResourceBytesToLoad() const { return m_totalPageAndResourceBytesToLoad; }
    long long totalBytesReceived() const { return m_totalBytesReceived; }

    bool isMainLoadProgressing() const;

private:
    void reset();
    void finalProgressComplete();
    void progressEstimateChanged(LocalFrame&);

    void progressHeartbeatTimerFired();
    Ref<Page> protectedPage() const;

    WeakRef<Page> m_page;
    UniqueRef<ProgressTrackerClient> m_client;
    WeakPtr<LocalFrame> m_originatingProgressFrame;
    HashMap<ResourceLoaderIdentifier, std::unique_ptr<ProgressItem>> m_progressItems;
    Timer m_progressHeartbeatTimer;

    long long m_totalPageAndResourceBytesToLoad { 0 };
    long long m_totalBytesReceived { 0 };
    long long m_totalBytesReceivedBeforePreviousHeartbeat { 0 };

    double m_lastNotifiedProgressValue { 0 };
    double m_progressValue { 0 };

    MonotonicTime m_mainLoadCompletionTime;
    MonotonicTime m_lastNotifiedProgressTime;

    int m_numProgressTrackedFrames { 0 };
    unsigned m_heartbeatsWithNoProgress { 0 };

    bool m_finalProgressChangedSent { false };
    bool m_isMainLoad { false };
};

} // namespace WebCore
