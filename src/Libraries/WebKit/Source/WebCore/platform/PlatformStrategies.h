/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 24, 2022.
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

#include <mutex>

namespace WebCore {

class BlobRegistry;
class LoaderStrategy;
class MediaStrategy;
class PasteboardStrategy;
#if ENABLE(DECLARATIVE_WEB_PUSH)
class PushStrategy;
#endif

class PlatformStrategies {
public:
    LoaderStrategy* loaderStrategy()
    {
        if (!m_loaderStrategy)
            m_loaderStrategy = createLoaderStrategy();
        return m_loaderStrategy;
    }

    PasteboardStrategy* pasteboardStrategy()
    {
        if (!m_pasteboardStrategy)
            m_pasteboardStrategy = createPasteboardStrategy();
        return m_pasteboardStrategy;
    }

    MediaStrategy& mediaStrategy()
    {
        std::call_once(m_onceKeyForMediaStrategies, [&] {
            m_mediaStrategy = createMediaStrategy();
        });
        return *m_mediaStrategy;
    }

    BlobRegistry* blobRegistry()
    {
        if (!m_blobRegistry)
            m_blobRegistry = createBlobRegistry();
        return m_blobRegistry;
    }

#if ENABLE(DECLARATIVE_WEB_PUSH)
    PushStrategy* pushStrategy()
    {
        if (!m_pushStrategy)
            m_pushStrategy = createPushStrategy();
        return m_pushStrategy;
    }
#endif

protected:
    PlatformStrategies() = default;

    virtual ~PlatformStrategies()
    {
    }

private:
    virtual LoaderStrategy* createLoaderStrategy() = 0;
    virtual PasteboardStrategy* createPasteboardStrategy() = 0;
    virtual MediaStrategy* createMediaStrategy() = 0;
    virtual BlobRegistry* createBlobRegistry() = 0;

    LoaderStrategy* m_loaderStrategy { };
    PasteboardStrategy* m_pasteboardStrategy { };
    std::once_flag m_onceKeyForMediaStrategies;
    MediaStrategy* m_mediaStrategy { };
    BlobRegistry* m_blobRegistry { };

#if ENABLE(DECLARATIVE_WEB_PUSH)
    virtual PushStrategy* createPushStrategy() = 0;
    PushStrategy* m_pushStrategy { };
#endif
};

bool hasPlatformStrategies();
WEBCORE_EXPORT PlatformStrategies* platformStrategies();
WEBCORE_EXPORT void setPlatformStrategies(PlatformStrategies*);
    
} // namespace WebCore
