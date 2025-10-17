/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 28, 2023.
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

#if ENABLE(WEB_AUDIO)

#include "HRTFDatabaseLoader.h"

#include "HRTFDatabase.h"
#include <wtf/HashMap.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

// Keeps track of loaders on a per-sample-rate basis.
static UncheckedKeyHashMap<double, HRTFDatabaseLoader*>& loaderMap()
{
    static NeverDestroyed<UncheckedKeyHashMap<double, HRTFDatabaseLoader*>> loaderMap;
    return loaderMap;
}

Ref<HRTFDatabaseLoader> HRTFDatabaseLoader::createAndLoadAsynchronouslyIfNecessary(float sampleRate)
{
    ASSERT(isMainThread());

    if (RefPtr<HRTFDatabaseLoader> loader = loaderMap().get(sampleRate)) {
        ASSERT(sampleRate == loader->databaseSampleRate());
        return loader.releaseNonNull();
    }

    auto loader = adoptRef(*new HRTFDatabaseLoader(sampleRate));
    loaderMap().add(sampleRate, loader.ptr());

    loader->loadAsynchronously();

    return loader;
}

HRTFDatabaseLoader::HRTFDatabaseLoader(float sampleRate)
    : m_databaseSampleRate(sampleRate)
{
    ASSERT(isMainThread());
}

HRTFDatabaseLoader::~HRTFDatabaseLoader()
{
    ASSERT(isMainThread());

    waitForLoaderThreadCompletion();
    m_hrtfDatabase = nullptr;

    // Remove ourself from the map.
    loaderMap().remove(m_databaseSampleRate);
}

void HRTFDatabaseLoader::load()
{
    ASSERT(!isMainThread());
    if (!m_hrtfDatabase.get()) {
        // Load the default HRTF database.
        m_hrtfDatabase = makeUnique<HRTFDatabase>(m_databaseSampleRate);
    }
}

void HRTFDatabaseLoader::loadAsynchronously()
{
    ASSERT(isMainThread());

    Locker locker { m_threadLock };
    
    if (!m_hrtfDatabase.get() && !m_databaseLoaderThread) {
        // Start the asynchronous database loading process.
        m_databaseLoaderThread = Thread::create("HRTF database loader"_s, [this] {
            load();
        });
    }
}

bool HRTFDatabaseLoader::isLoaded() const
{
    return m_hrtfDatabase.get();
}

void HRTFDatabaseLoader::waitForLoaderThreadCompletion()
{
    Locker locker { m_threadLock };
    
    // waitForThreadCompletion() should not be called twice for the same thread.
    if (m_databaseLoaderThread)
        m_databaseLoaderThread->waitForCompletion();
    m_databaseLoaderThread = nullptr;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
