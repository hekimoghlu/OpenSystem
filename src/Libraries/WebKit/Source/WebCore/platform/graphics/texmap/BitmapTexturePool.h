/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 11, 2023.
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

#if USE(TEXTURE_MAPPER)

#include "BitmapTexture.h"
#include <wtf/RunLoop.h>

namespace WebCore {
class BitmapTexturePool;
}

namespace WTF {
template<typename T> struct IsDeprecatedTimerSmartPointerException;
template<> struct IsDeprecatedTimerSmartPointerException<WebCore::BitmapTexturePool> : std::true_type { };
}

namespace WebCore {

class IntSize;

class BitmapTexturePool {
    WTF_MAKE_NONCOPYABLE(BitmapTexturePool);
    WTF_MAKE_FAST_ALLOCATED();
public:
    BitmapTexturePool();

    Ref<BitmapTexture> acquireTexture(const IntSize&, OptionSet<BitmapTexture::Flags>);
    void releaseUnusedTexturesTimerFired();

private:
    struct Entry {
        explicit Entry(Ref<BitmapTexture>&& texture)
            : m_texture(WTFMove(texture))
        { }

        void markIsInUse() { m_lastUsedTime = MonotonicTime::now(); }
        bool canBeReleased (MonotonicTime minUsedTime) const { return m_lastUsedTime < minUsedTime && m_texture->refCount() == 1; }

        Ref<BitmapTexture> m_texture;
        MonotonicTime m_lastUsedTime;
    };

    void scheduleReleaseUnusedTextures();
    void enterLimitExceededModeIfNeeded();
    void exitLimitExceededModeIfNeeded();

    Vector<Entry> m_textures;
    RunLoop::Timer m_releaseUnusedTexturesTimer;
    uint64_t m_poolSize { 0 };
    bool m_onLimitExceededMode { false };
    Seconds m_releaseUnusedSecondsTolerance;
    Seconds m_releaseUnusedTexturesTimerInterval;
};

} // namespace WebCore

#endif // USE(TEXTURE_MAPPER)
