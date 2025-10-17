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

#include <wtf/FastMalloc.h>
#include <wtf/ThreadGroup.h>

#if USE(GLIB)
#include <optional>
#include <wtf/RunLoop.h>
#include <wtf/glib/GRefPtr.h>

typedef struct _GDBusProxy GDBusProxy;
#endif

namespace WTF {

class RealTimeThreads {
    WTF_MAKE_FAST_ALLOCATED;
    friend class LazyNeverDestroyed<RealTimeThreads>;
public:
    WTF_EXPORT_PRIVATE static RealTimeThreads& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    void registerThread(Thread&);

    WTF_EXPORT_PRIVATE void setEnabled(bool);

private:
    RealTimeThreads();

    void promoteThreadToRealTime(const WTF::Thread&);
    void demoteThreadFromRealTime(const WTF::Thread&);
    void demoteAllThreadsFromRealTime();

#if USE(GLIB)
    void realTimeKitMakeThreadRealTime(uint64_t processID, uint64_t threadID, uint32_t priority);
    void scheduleDiscardRealTimeKitProxy();
    void discardRealTimeKitProxyTimerFired();
#endif

    std::shared_ptr<ThreadGroup> m_threadGroup;
    bool m_enabled { true };
#if USE(GLIB)
    std::optional<GRefPtr<GDBusProxy>> m_realTimeKitProxy;
    RunLoop::Timer m_discardRealTimeKitProxyTimer;
#endif
};

} // namespace WTF

using WTF::RealTimeThreads;
