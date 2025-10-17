/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#include "CoordinatedPlatformLayerBufferProxy.h"

#if USE(COORDINATED_GRAPHICS)
#include "CoordinatedPlatformLayer.h"
#include "CoordinatedPlatformLayerBuffer.h"
#include <wtf/threads/BinarySemaphore.h>

namespace WebCore {

Ref<CoordinatedPlatformLayerBufferProxy> CoordinatedPlatformLayerBufferProxy::create()
{
    return adoptRef(*new CoordinatedPlatformLayerBufferProxy());
}

CoordinatedPlatformLayerBufferProxy::CoordinatedPlatformLayerBufferProxy() = default;

CoordinatedPlatformLayerBufferProxy::~CoordinatedPlatformLayerBufferProxy()
{
    ASSERT(!m_layer);
#if ENABLE(VIDEO) && USE(GSTREAMER)
    ASSERT(!m_compositingRunLoop);
#endif
}

void CoordinatedPlatformLayerBufferProxy::setTargetLayer(CoordinatedPlatformLayer* layer)
{
    ASSERT(RunLoop::isMain());
    Locker locker { m_lock };
    if (m_layer == layer)
        return;

    m_layer = layer;
    if (m_layer) {
        m_isValid = true;
#if ENABLE(VIDEO) && USE(GSTREAMER)
        m_compositingRunLoop = m_layer->compositingRunLoop();
#endif
    } else {
        m_isValid = false;
        m_pendingBuffer = nullptr;
#if ENABLE(VIDEO) && USE(GSTREAMER)
        m_compositingRunLoop = nullptr;
#endif
    }
}

void CoordinatedPlatformLayerBufferProxy::consumePendingBufferIfNeeded()
{
    ASSERT(RunLoop::isMain());
    Locker locker { m_lock };
    if (!m_pendingBuffer)
        return;

    if (m_layer)
        m_layer->setContentsBuffer(WTFMove(m_pendingBuffer));
    else
        m_pendingBuffer = nullptr;
}

bool CoordinatedPlatformLayerBufferProxy::setDisplayBuffer(std::unique_ptr<CoordinatedPlatformLayerBuffer>&& buffer)
{
    Locker locker { m_lock };
    if (!m_isValid)
        return false;

    if (!m_layer) {
        m_pendingBuffer = WTFMove(buffer);
        return true;
    }

    m_pendingBuffer = nullptr;

    {
        Locker layerLocker { m_layer->lock() };
        m_layer->setContentsBuffer(WTFMove(buffer), CoordinatedPlatformLayer::RequireComposition::No);
    }
    m_layer->requestComposition();

    return true;
}

#if ENABLE(VIDEO) && USE(GSTREAMER)
void CoordinatedPlatformLayerBufferProxy::dropCurrentBufferWhilePreservingTexture(ShouldWait shouldWait)
{
    RefPtr<RunLoop> compositingRunLoop;
    {
        Locker locker { m_lock };
        if (!m_isValid || !m_layer || !m_compositingRunLoop)
            return;

        compositingRunLoop = m_compositingRunLoop;
    }

    auto dropCurrentBuffer = [this, protectedThis = Ref { *this }] {
        Locker locker { m_lock };
        if (!m_isValid || !m_layer)
            return;

        m_layer->replaceCurrentContentsBufferWithCopy();
    };

    if (shouldWait == ShouldWait::No) {
        compositingRunLoop->dispatch(WTFMove(dropCurrentBuffer));
        return;
    }

    BinarySemaphore semaphore;
    compositingRunLoop->dispatch([&semaphore, function = WTFMove(dropCurrentBuffer)]() mutable {
        function();
        semaphore.signal();
    });
    semaphore.wait();
}
#endif

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
