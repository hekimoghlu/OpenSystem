/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#include "config.h"
#include "ThreadGlobalData.h"

#include "CachedResourceRequestInitiatorTypes.h"
#include "EventNames.h"
#include "FontCache.h"
#include "MIMETypeRegistry.h"
#include "QualifiedNameCache.h"
#include "ThreadTimers.h"
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Threading.h>
#include <wtf/text/StringImpl.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ThreadGlobalData);

ThreadGlobalData::ThreadGlobalData()
    : m_threadTimers(makeUnique<ThreadTimers>())
#ifndef NDEBUG
    , m_isMainThread(isMainThread())
#endif
{
}

ThreadGlobalData::~ThreadGlobalData() = default;

void ThreadGlobalData::destroy()
{
    if (m_fontCache)
        m_fontCache->invalidate();
    m_fontCache = nullptr;
    m_destroyed = true;
}

#if USE(WEB_THREAD)
static ThreadGlobalData* sharedMainThreadStaticData { nullptr };

void ThreadGlobalData::setWebCoreThreadData()
{
    ASSERT(isWebThread());
    ASSERT(&threadGlobalData() != sharedMainThreadStaticData);

    // Set WebThread's ThreadGlobalData object to be the same as the main UI thread.
    Thread::current().m_clientData = adoptRef(sharedMainThreadStaticData);

    ASSERT(&threadGlobalData() == sharedMainThreadStaticData);
}

ThreadGlobalData& threadGlobalDataSlow()
{
    auto& thread = Thread::current();
    auto* clientData = thread.m_clientData.get();
    if (UNLIKELY(clientData))
        return *static_cast<ThreadGlobalData*>(clientData);

    auto data = adoptRef(*new ThreadGlobalData);
    if (pthread_main_np()) {
        sharedMainThreadStaticData = data.ptr();
        data->ref();
    }

    clientData = data.ptr();
    thread.m_clientData = WTFMove(data);
    return *static_cast<ThreadGlobalData*>(clientData);
}

#else

ThreadGlobalData& threadGlobalDataSlow()
{
    auto& thread = Thread::current();
    auto* clientData = thread.m_clientData.get();
    if (UNLIKELY(clientData))
        return *static_cast<ThreadGlobalData*>(clientData);

    auto data = adoptRef(*new ThreadGlobalData);
    clientData = data.ptr();
    thread.m_clientData = WTFMove(data);
    return *static_cast<ThreadGlobalData*>(clientData);
}

#endif

void ThreadGlobalData::initializeCachedResourceRequestInitiatorTypes()
{
    ASSERT(!m_cachedResourceRequestInitiatorTypes);
    m_cachedResourceRequestInitiatorTypes = makeUnique<CachedResourceRequestInitiatorTypes>();
}

void ThreadGlobalData::initializeEventNames()
{
    ASSERT(!m_eventNames);
    m_eventNames = EventNames::create();
}

void ThreadGlobalData::initializeQualifiedNameCache()
{
    ASSERT(!m_qualifiedNameCache);
    m_qualifiedNameCache = makeUnique<QualifiedNameCache>();
}

void ThreadGlobalData::initializeMimeTypeRegistryThreadGlobalData()
{
    ASSERT(!m_MIMETypeRegistryThreadGlobalData);
    m_MIMETypeRegistryThreadGlobalData = MIMETypeRegistry::createMIMETypeRegistryThreadGlobalData();
}

void ThreadGlobalData::initializeFontCache()
{
    ASSERT(!m_fontCache);
    m_fontCache = makeUnique<FontCache>();
}

} // namespace WebCore

namespace PAL {

ThreadGlobalData& threadGlobalData()
{
    return WebCore::threadGlobalData();
}

} // namespace PAL
