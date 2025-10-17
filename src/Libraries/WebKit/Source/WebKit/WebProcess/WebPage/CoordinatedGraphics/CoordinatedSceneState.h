/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 8, 2023.
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
#include <atomic>
#include <wtf/HashSet.h>
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {
class CoordinatedPlatformLayer;
}

namespace WebKit {

class CoordinatedSceneState final : public ThreadSafeRefCounted<CoordinatedSceneState> {
    WTF_MAKE_TZONE_ALLOCATED(CoordinatedSceneState);
public:
    static Ref<CoordinatedSceneState> create()
    {
        return adoptRef(*new CoordinatedSceneState());
    }
    virtual ~CoordinatedSceneState();

    WebCore::CoordinatedPlatformLayer& rootLayer() const { return m_rootLayer.get(); }

    void setRootLayerChildren(Vector<Ref<WebCore::CoordinatedPlatformLayer>>&&);
    void addLayer(WebCore::CoordinatedPlatformLayer&);
    void removeLayer(WebCore::CoordinatedPlatformLayer&);

    bool flush();
    void invalidate();

    const HashSet<Ref<WebCore::CoordinatedPlatformLayer>>& committedLayers();
    void invalidateCommittedLayers();

#if !HAVE(DISPLAY_LINK)
    bool layersDidChange() const { return m_didChangeLayers; }
#endif

    void waitUntilPaintingComplete();

private:
    CoordinatedSceneState();

    Ref<WebCore::CoordinatedPlatformLayer> m_rootLayer;
    HashSet<Ref<WebCore::CoordinatedPlatformLayer>> m_layers;
    Lock m_pendingLayersLock;
    HashSet<Ref<WebCore::CoordinatedPlatformLayer>> m_pendingLayers WTF_GUARDED_BY_LOCK(m_pendingLayersLock);
    std::atomic<bool> m_didChangeLayers { false };
    HashSet<Ref<WebCore::CoordinatedPlatformLayer>> m_committedLayers;
};

} // namespace WebKit

#endif // USE(COORDINATED_GRAPHICS)

