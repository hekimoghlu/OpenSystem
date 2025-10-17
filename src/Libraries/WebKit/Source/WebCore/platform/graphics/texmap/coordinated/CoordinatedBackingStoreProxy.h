/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
#include "FloatPoint.h"
#include "IntPoint.h"
#include "IntPointHash.h"
#include "IntRect.h"
#include <wtf/Assertions.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {
class CoordinatedPlatformLayer;
class CoordinatedTileBuffer;
class GraphicsLayer;

class CoordinatedBackingStoreProxy final : public ThreadSafeRefCounted<CoordinatedBackingStoreProxy> {
    WTF_MAKE_TZONE_ALLOCATED(CoordinatedBackingStoreProxy);
public:
    static Ref<CoordinatedBackingStoreProxy> create(float contentsScale, std::optional<IntSize> tileSize = std::nullopt);
    ~CoordinatedBackingStoreProxy();

    bool setContentsScale(float);
    const IntRect& coverRect() const { return m_coverRect; }

    class Update {
        WTF_MAKE_NONCOPYABLE(Update);
    public:
        Update() = default;
        Update(Update&&) = default;
        Update& operator=(Update&&) = default;
        ~Update();

        struct TileUpdate {
            uint32_t tileID { 0 };
            IntRect tileRect;
            IntRect dirtyRect;
            Ref<CoordinatedTileBuffer> buffer;
        };

        float scale() const { return m_scale; }
        const Vector<uint32_t>& tilesToCreate() const { return m_tilesToCreate; }
        const Vector<TileUpdate>& tilesToUpdate() const { return m_tilesToUpdate; }
        const Vector<uint32_t>& tilesToRemove() const { return m_tilesToRemove; }

        void appendUpdate(float, Vector<uint32_t>&&, Vector<TileUpdate>&&, Vector<uint32_t>&&);
        void waitUntilPaintingComplete();

    private:
        float m_scale { 1 };
        Vector<uint32_t> m_tilesToCreate;
        Vector<TileUpdate> m_tilesToUpdate;
        Vector<uint32_t> m_tilesToRemove;
    };

    enum class UpdateResult : uint8_t {
        BuffersChanged = 1 << 0,
        TilesPending = 1 << 1,
        TilesChanged = 1 << 2
    };
    OptionSet<UpdateResult> updateIfNeeded(const IntRect& unscaledVisibleRect, const IntRect& unscaledContentsRect, bool shouldCreateAndDestroyTiles, const Vector<IntRect, 1>&, CoordinatedPlatformLayer&);
    Update takePendingUpdate();

    void waitUntilPaintingComplete();

private:
    struct Tile {
        Tile() = default;
        Tile(uint32_t id, const IntPoint& position, IntRect&& tileRect)
            : id(id)
            , position(position)
            , rect(WTFMove(tileRect))
            , dirtyRect(rect)
        {
        }
        Tile(const Tile&) = delete;
        Tile& operator=(const Tile&) = delete;
        Tile(Tile&&) = default;
        Tile& operator=(Tile&&) = default;

        void resize(const IntSize& size)
        {
            rect.setSize(size);
            dirtyRect = rect;
        }

        void addDirtyRect(const IntRect& dirty)
        {
            auto tileDirtyRect = intersection(dirty, rect);
            dirtyRect.unite(tileDirtyRect);
        }

        bool isDirty() const
        {
            return !dirtyRect.isEmpty();
        }

        void markClean()
        {
            dirtyRect = { };
        }

        uint32_t id { 0 };
        IntPoint position;
        IntRect rect;
        IntRect dirtyRect;
    };

    CoordinatedBackingStoreProxy(float contentsScale, const IntSize& tileSize);

    void invalidateRegion(const Vector<IntRect, 1>&);
    void createOrDestroyTiles(const IntRect& visibleRect, const IntRect& scaledContentsRect, float coverAreaMultiplier, Vector<uint32_t>& tilesToCreate, Vector<uint32_t>& tilesToRemove);
    std::pair<IntRect, IntRect> computeCoverAndKeepRect() const;

    void adjustForContentsRect(IntRect&) const;

    IntRect mapToContents(const IntRect&) const;
    IntRect mapFromContents(const IntRect&) const;
    IntRect tileRectForPosition(const IntPoint&) const;
    IntPoint tilePositionForPoint(const IntPoint&) const;
    void forEachTilePositionInRect(const IntRect&, Function<void(IntPoint&&)>&&);

    float m_contentsScale { 1 };
    IntSize m_tileSize;
    float m_coverAreaMultiplier { 2 };
    bool m_pendingTileCreation { false };
    IntRect m_contentsRect;
    IntRect m_visibleRect;
    IntRect m_coverRect;
    IntRect m_keepRect;
    UncheckedKeyHashMap<IntPoint, Tile> m_tiles;
    struct {
        Lock lock;
        Update pending WTF_GUARDED_BY_LOCK(lock);
    } m_update;
};

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS)
