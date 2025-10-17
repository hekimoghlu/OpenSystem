/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

#if USE(COORDINATED_GRAPHICS)
#include <wtf/Lock.h>
#include <wtf/RefPtr.h>
#include <wtf/RunLoop.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {
class CoordinatedPlatformLayer;
class CoordinatedPlatformLayerBuffer;
class IntSize;
class TextureMapperLayer;

class CoordinatedPlatformLayerBufferProxy final : public ThreadSafeRefCounted<CoordinatedPlatformLayerBufferProxy> {
public:
    static Ref<CoordinatedPlatformLayerBufferProxy> create();
    virtual ~CoordinatedPlatformLayerBufferProxy();

    void setTargetLayer(CoordinatedPlatformLayer*);
    void consumePendingBufferIfNeeded();
    bool setDisplayBuffer(std::unique_ptr<CoordinatedPlatformLayerBuffer>&&);

#if ENABLE(VIDEO) && USE(GSTREAMER)
    enum class ShouldWait : bool { No, Yes };
    void dropCurrentBufferWhilePreservingTexture(ShouldWait);
#endif

private:
    CoordinatedPlatformLayerBufferProxy();

    Lock m_lock;
    bool m_isValid WTF_GUARDED_BY_LOCK(m_lock) { true };
    RefPtr<CoordinatedPlatformLayer> m_layer WTF_GUARDED_BY_LOCK(m_lock);
    std::unique_ptr<CoordinatedPlatformLayerBuffer> m_pendingBuffer WTF_GUARDED_BY_LOCK(m_lock);
#if ENABLE(VIDEO) && USE(GSTREAMER)
    RefPtr<RunLoop> m_compositingRunLoop WTF_GUARDED_BY_LOCK(m_lock);
#endif
};

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
