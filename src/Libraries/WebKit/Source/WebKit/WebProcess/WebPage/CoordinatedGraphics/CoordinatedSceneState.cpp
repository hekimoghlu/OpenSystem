/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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
#include "CoordinatedSceneState.h"

#if USE(COORDINATED_GRAPHICS)
#include <WebCore/CoordinatedPlatformLayer.h>
#include <wtf/MainThread.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(CoordinatedSceneState);

CoordinatedSceneState::CoordinatedSceneState()
    : m_rootLayer(CoordinatedPlatformLayer::create())
{
    ASSERT(isMainRunLoop());
}

CoordinatedSceneState::~CoordinatedSceneState()
{
    ASSERT(m_layers.isEmpty());
    ASSERT(m_pendingLayers.isEmpty());
    ASSERT(m_committedLayers.isEmpty());
}

void CoordinatedSceneState::setRootLayerChildren(Vector<Ref<CoordinatedPlatformLayer>>&& children)
{
    ASSERT(isMainRunLoop());

    {
        Locker locker { m_rootLayer->lock() };
        m_rootLayer->setChildren(WTFMove(children));
    }
    m_didChangeLayers = true;
}

void CoordinatedSceneState::addLayer(CoordinatedPlatformLayer& layer)
{
    ASSERT(isMainRunLoop());
    m_layers.add(layer);
    m_didChangeLayers = true;
}

void CoordinatedSceneState::removeLayer(CoordinatedPlatformLayer& layer)
{
    ASSERT(isMainRunLoop());
    m_layers.remove(layer);
    m_didChangeLayers = true;
}

bool CoordinatedSceneState::flush()
{
    ASSERT(isMainRunLoop());
    if (!m_didChangeLayers)
        return false;

    m_didChangeLayers = false;

    Locker pendingLayersLock { m_pendingLayersLock };
    m_pendingLayers = m_layers;
    return true;
}

const HashSet<Ref<CoordinatedPlatformLayer>>& CoordinatedSceneState::committedLayers()
{
    ASSERT(!isMainRunLoop());
    Locker pendingLayersLock { m_pendingLayersLock };
    if (!m_pendingLayers.isEmpty()) {
        auto removedLayers = m_committedLayers.differenceWith(m_pendingLayers);
        m_committedLayers = WTFMove(m_pendingLayers);
        for (auto& layer : removedLayers)
            layer->invalidateTarget();
    }
    return m_committedLayers;
}

void CoordinatedSceneState::invalidateCommittedLayers()
{
    ASSERT(!isMainRunLoop());
    m_rootLayer->invalidateTarget();
    while (!m_committedLayers.isEmpty()) {
        auto layer = m_committedLayers.takeAny();
        layer->invalidateTarget();
    }
}

void CoordinatedSceneState::invalidate()
{
    ASSERT(isMainRunLoop());
    // Root layer doesn't have client nor backing stores to invalidate.
    while (!m_layers.isEmpty()) {
        auto layer = m_layers.takeAny();
        layer->invalidateClient();
    }

    Locker pendingLayersLock { m_pendingLayersLock };
    m_pendingLayers = { };
}

void CoordinatedSceneState::waitUntilPaintingComplete()
{
    ASSERT(isMainRunLoop());
    Locker pendingLayersLock { m_pendingLayersLock };
    for (auto& layer : m_pendingLayers)
        layer->waitUntilPaintingComplete();
}

} // namespace WebKit

#endif // USE(COORDINATED_GRAPHICS)
