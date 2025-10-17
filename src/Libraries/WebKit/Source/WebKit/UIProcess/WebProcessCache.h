/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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

#include "WebProcessProxy.h"
#include <WebCore/Site.h>
#include <pal/SessionID.h>
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class ProcessThrottlerActivity;
class WebProcessPool;
class WebsiteDataStore;

class WebProcessCache final : public CanMakeCheckedPtr<WebProcessCache> {
    WTF_MAKE_TZONE_ALLOCATED(WebProcessCache);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebProcessCache);
public:
    explicit WebProcessCache(WebProcessPool&);

    bool addProcessIfPossible(Ref<WebProcessProxy>&&);
    RefPtr<WebProcessProxy> takeProcess(const WebCore::Site&, WebsiteDataStore&, WebProcessProxy::LockdownMode, const API::PageConfiguration&);

    void updateCapacity(WebProcessPool&);
    unsigned capacity() const { return m_capacity; }

    unsigned size() const { return m_processesPerSite.size(); }

    void clear();
    void setApplicationIsActive(bool);

    void clearAllProcessesForSession(PAL::SessionID);

    enum class ShouldShutDownProcess : bool { No, Yes };
    void removeProcess(WebProcessProxy&, ShouldShutDownProcess);
    static void setCachedProcessSuspensionDelayForTesting(Seconds);

private:
    static Seconds cachedProcessLifetime;
    static Seconds clearingDelayAfterApplicationResignsActive;

    class CachedProcess : public RefCounted<CachedProcess> {
        WTF_MAKE_TZONE_ALLOCATED(CachedProcess);
    public:
        static Ref<CachedProcess> create(Ref<WebProcessProxy>&&);
        ~CachedProcess();

        Ref<WebProcessProxy> takeProcess();
        WebProcessProxy& process() { ASSERT(m_process); return *m_process; }
        void startSuspensionTimer();

#if PLATFORM(MAC) || PLATFORM(GTK) || PLATFORM(WPE)
        bool isSuspended() const { return !m_suspensionTimer.isActive(); }
#endif

    private:
        explicit CachedProcess(Ref<WebProcessProxy>&&);

        void evictionTimerFired();
#if PLATFORM(MAC) || PLATFORM(GTK) || PLATFORM(WPE)
        void suspensionTimerFired();
#endif

        RefPtr<WebProcessProxy> m_process;
        RunLoop::Timer m_evictionTimer;
#if PLATFORM(MAC) || PLATFORM(GTK) || PLATFORM(WPE)
        RunLoop::Timer m_suspensionTimer;
        RefPtr<ProcessThrottlerActivity> m_backgroundActivity;
#endif
    };

    bool canCacheProcess(WebProcessProxy&) const;
    void platformInitialize();
    bool addProcess(Ref<CachedProcess>&&);

    unsigned m_capacity { 0 };

    HashMap<uint64_t, Ref<CachedProcess>> m_pendingAddRequests;
    HashMap<WebCore::Site, Ref<CachedProcess>> m_processesPerSite;
    RunLoop::Timer m_evictionTimer;
};

} // namespace WebKit
