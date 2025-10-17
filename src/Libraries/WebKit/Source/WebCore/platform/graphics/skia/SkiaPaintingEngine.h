/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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

#if USE(COORDINATED_GRAPHICS) && USE(SKIA)
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WorkerPool.h>

namespace WebCore {
class BitmapTexturePool;
class CoordinatedTileBuffer;
class GraphicsContext;
class GraphicsLayer;
class IntRect;
class IntSize;
enum class RenderingMode : uint8_t;

namespace DisplayList {
class DisplayList;
}

class SkiaPaintingEngine {
    WTF_MAKE_TZONE_ALLOCATED(SkiaPaintingEngine);
    WTF_MAKE_NONCOPYABLE(SkiaPaintingEngine);
public:
    SkiaPaintingEngine(unsigned numberOfCPUThreads, unsigned numberOfGPUThreads);
    ~SkiaPaintingEngine();

    static std::unique_ptr<SkiaPaintingEngine> create();

    static unsigned numberOfCPUPaintingThreads();
    static unsigned numberOfGPUPaintingThreads();

    Ref<CoordinatedTileBuffer> paintLayer(const GraphicsLayer&, const IntRect& dirtyRect, bool contentsOpaque, float contentsScale);

private:
    Ref<CoordinatedTileBuffer> createBuffer(RenderingMode, const IntSize&, bool contentsOpaque) const;
    std::unique_ptr<DisplayList::DisplayList> recordDisplayList(RenderingMode, const GraphicsLayer&, const IntRect& dirtyRect, bool contentsOpaque, float contentsScale) const;
    void paintIntoGraphicsContext(const GraphicsLayer&, GraphicsContext&, const IntRect&, bool contentsOpaque, float contentsScale) const;

    static bool paintDisplayListIntoBuffer(Ref<CoordinatedTileBuffer>&, DisplayList::DisplayList&);
    bool paintGraphicsLayerIntoBuffer(Ref<CoordinatedTileBuffer>&, const GraphicsLayer&, const IntRect& dirtyRect, bool contentsOpaque, float contentsScale) const;

    // Threaded rendering
    Ref<CoordinatedTileBuffer> postPaintingTask(const GraphicsLayer&, RenderingMode, const IntRect& dirtyRect, bool contentsOpaque, float contentsScale);

    // Main thread rendering
    Ref<CoordinatedTileBuffer> performPaintingTask(const GraphicsLayer&, RenderingMode, const IntRect& dirtyRect, bool contentsOpaque, float contentsScale);

    RenderingMode renderingMode() const;
    std::optional<RenderingMode> threadedRenderingMode() const;

    RefPtr<WorkerPool> m_cpuWorkerPool;
    RefPtr<WorkerPool> m_gpuWorkerPool;
    std::unique_ptr<BitmapTexturePool> m_texturePool;
};

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS) && USE(SKIA)
