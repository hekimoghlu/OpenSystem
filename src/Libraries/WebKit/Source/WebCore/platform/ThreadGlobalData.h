/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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

#include <pal/ThreadGlobalData.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/StringHash.h>

namespace JSC {
class CallFrame;
class JSGlobalObject;
}

namespace WebCore {

class FontCache;
class QualifiedNameCache;
class ThreadTimers;

struct CachedResourceRequestInitiatorTypes;
struct EventNames;
struct MIMETypeRegistryThreadGlobalData;

class ThreadGlobalData : public PAL::ThreadGlobalData {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ThreadGlobalData, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(ThreadGlobalData);
public:
    WEBCORE_EXPORT ThreadGlobalData();
    WEBCORE_EXPORT ~ThreadGlobalData();
    void destroy(); // called on workers to clean up the ThreadGlobalData before the thread exits.

    const CachedResourceRequestInitiatorTypes& cachedResourceRequestInitiatorTypes()
    {
        ASSERT(!m_destroyed);
        if (UNLIKELY(!m_cachedResourceRequestInitiatorTypes))
            initializeCachedResourceRequestInitiatorTypes();
        return *m_cachedResourceRequestInitiatorTypes;
    }
    EventNames& eventNames()
    {
        ASSERT(!m_destroyed);
        if (UNLIKELY(!m_eventNames))
            initializeEventNames();
        return *m_eventNames;
    }
    QualifiedNameCache& qualifiedNameCache()
    {
        ASSERT(!m_destroyed);
        if (UNLIKELY(!m_qualifiedNameCache))
            initializeQualifiedNameCache();
        return *m_qualifiedNameCache;
    }
    const MIMETypeRegistryThreadGlobalData& mimeTypeRegistryThreadGlobalData()
    {
        ASSERT(!m_destroyed);
        if (UNLIKELY(!m_MIMETypeRegistryThreadGlobalData))
            initializeMimeTypeRegistryThreadGlobalData();
        return *m_MIMETypeRegistryThreadGlobalData;
    }

    ThreadTimers& threadTimers() { return *m_threadTimers; }

    JSC::JSGlobalObject* currentState() const { return m_currentState; }
    void setCurrentState(JSC::JSGlobalObject* state) { m_currentState = state; }

#if USE(WEB_THREAD)
    void setWebCoreThreadData();
#endif

    bool isInRemoveAllEventListeners() const { return m_isInRemoveAllEventListeners; }
    void setIsInRemoveAllEventListeners(bool value) { m_isInRemoveAllEventListeners = value; }

    FontCache& fontCache()
    {
        ASSERT(!m_destroyed);
        if (UNLIKELY(!m_fontCache))
            initializeFontCache();
        return *m_fontCache;
    }

    FontCache* fontCacheIfExists() { return m_fontCache.get(); }
    FontCache* fontCacheIfNotDestroyed() { return m_destroyed ? nullptr : &fontCache(); }

private:
    bool m_destroyed { false };

    WEBCORE_EXPORT void initializeCachedResourceRequestInitiatorTypes();
    WEBCORE_EXPORT void initializeEventNames();
    WEBCORE_EXPORT void initializeQualifiedNameCache();
    WEBCORE_EXPORT void initializeMimeTypeRegistryThreadGlobalData();
    WEBCORE_EXPORT void initializeFontCache();

    std::unique_ptr<CachedResourceRequestInitiatorTypes> m_cachedResourceRequestInitiatorTypes;
    std::unique_ptr<EventNames> m_eventNames;
    std::unique_ptr<ThreadTimers> m_threadTimers;
    std::unique_ptr<QualifiedNameCache> m_qualifiedNameCache;
    JSC::JSGlobalObject* m_currentState { nullptr };
    std::unique_ptr<MIMETypeRegistryThreadGlobalData> m_MIMETypeRegistryThreadGlobalData;
    std::unique_ptr<FontCache> m_fontCache;

#ifndef NDEBUG
    bool m_isMainThread;
#endif

    bool m_isInRemoveAllEventListeners { false };

    friend ThreadGlobalData& threadGlobalData();
};


#if USE(WEB_THREAD)
WEBCORE_EXPORT ThreadGlobalData& threadGlobalDataSlow();
#else
WEBCORE_EXPORT ThreadGlobalData& threadGlobalDataSlow() PURE_FUNCTION;
#endif

#if USE(WEB_THREAD)
inline ThreadGlobalData& threadGlobalData()
#else
inline PURE_FUNCTION ThreadGlobalData& threadGlobalData()
#endif
{
#if HAVE(FAST_TLS)
    if (auto* thread = Thread::currentMayBeNull(); LIKELY(thread)) {
        if (auto* clientData = thread->m_clientData.get(); LIKELY(clientData))
            return *static_cast<ThreadGlobalData*>(clientData);
    }
#else
    auto& thread = Thread::current();
    auto* clientData = thread.m_clientData.get();
    if (LIKELY(clientData))
        return *static_cast<ThreadGlobalData*>(clientData);
#endif
    return threadGlobalDataSlow();
}

} // namespace WebCore
