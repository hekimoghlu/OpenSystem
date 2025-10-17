/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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

#if USE(GRAPHICS_LAYER_WC)

#include <WebCore/IntRect.h>
#include <WebCore/TextureMapperSparseBackingStore.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WCTileGrid {
public:
    using TileIndex = WebCore::TextureMapperSparseBackingStore::TileIndex;

    class Tile {
        WTF_MAKE_TZONE_ALLOCATED(WCTileGrid);
        WTF_MAKE_NONCOPYABLE(Tile);
    public:
        Tile(WebCore::IntRect);
        bool willRemove() const { return m_willRemove; }
        void setWillRemove(bool v) { m_willRemove = v; }
        void addDirtyRect(const WebCore::IntRect&);
        void clearDirtyRect();
        bool hasDirtyRect() const;
        WebCore::IntRect& dirtyRect() { return m_dirtyRect; }

    private:
        bool m_willRemove { false };
        WebCore::IntRect m_tileRect;
        WebCore::IntRect m_dirtyRect;
    };

    void setSize(const WebCore::IntSize&);
    void addDirtyRect(const WebCore::IntRect&);
    void clearDirtyRects();
    bool setCoverageRect(const WebCore::IntRect&);
    auto& tiles() { return m_tiles; }
    WebCore::IntSize tilePixelSize() const;

private:
    bool ensureTile(TileIndex);
    WebCore::IntRect tileRectFromPixelRect(const WebCore::IntRect&);
    WebCore::IntSize tileSizeFromPixelSize(const WebCore::IntSize&);
    
    WebCore::IntSize m_size;
    HashMap<TileIndex, std::unique_ptr<Tile>> m_tiles;
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
