/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 25, 2022.
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

#if ENABLE(UNIFIED_PDF)

#include "PDFDocumentLayout.h"
#include "PDFPageCoverage.h"
#include <WebCore/FloatRect.h>
#include <WebCore/GraphicsLayer.h>
#include <WebCore/IntPoint.h>
#include <WebCore/TiledBacking.h>
#include <wtf/HashMap.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WorkQueue.h>

OBJC_CLASS PDFDocument;

namespace WebKit {

struct TileForGrid {
    WebCore::TileGridIdentifier gridIdentifier;
    WebCore::TileIndex tileIndex;

    bool operator==(const TileForGrid&) const = default;

    unsigned computeHash() const
    {
        return WTF::computeHash(gridIdentifier.toUInt64(), tileIndex.x(), tileIndex.y());
    }
};

WTF::TextStream& operator<<(WTF::TextStream&, const TileForGrid&);

struct PDFTileRenderType;
using PDFTileRenderIdentifier = ObjectIdentifier<PDFTileRenderType>;

struct TileRenderInfo {
    WebCore::FloatRect tileRect;
    WebCore::FloatRect renderRect; // Represents the portion of the tile that needs rendering (in the same coordinate system as tileRect).
    RefPtr<WebCore::NativeImage> background; // Optional existing content around renderRect, will be rendered to tileRect.
    PDFPageCoverageAndScales pageCoverage;
    bool showDebugIndicators { false };

    bool operator==(const TileRenderInfo&) const = default;
    bool equivalentForPainting(const TileRenderInfo& other) const
    {
        return tileRect == other.tileRect && pageCoverage == other.pageCoverage;
    }
};

WTF::TextStream& operator<<(WTF::TextStream&, const TileRenderInfo&);

struct TileRenderData {
    PDFTileRenderIdentifier renderIdentifier;
    TileRenderInfo renderInfo;
};

WTF::TextStream& operator<<(WTF::TextStream&, const TileRenderData&);

struct PagePreviewRequest {
    PDFDocumentLayout::PageIndex pageIndex;
    WebCore::FloatRect normalizedPageBounds;
    float scale { 1.0f };
    bool showDebugIndicators { false };
};

} // namespace WebKit

namespace WTF {

struct TileForGridHash {
    static unsigned hash(const WebKit::TileForGrid& key)
    {
        return key.computeHash();
    }
    static bool equal(const WebKit::TileForGrid& a, const WebKit::TileForGrid& b)
    {
        return a == b;
    }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<WebKit::TileForGrid> : GenericHashTraits<WebKit::TileForGrid> {
    static constexpr bool emptyValueIsZero = true;
    static WebKit::TileForGrid emptyValue() { return { HashTraits<WebCore::TileGridIdentifier>::emptyValue(), { 0, 0 } }; }
    static bool isEmptyValue(const WebKit::TileForGrid& value) { return value.gridIdentifier.isHashTableEmptyValue(); }
    static void constructDeletedValue(WebKit::TileForGrid& tileForGrid) { HashTraits<WebCore::TileGridIdentifier>::constructDeletedValue(tileForGrid.gridIdentifier); }
    static bool isDeletedValue(const WebKit::TileForGrid& tileForGrid) { return tileForGrid.gridIdentifier.isHashTableDeletedValue(); }
};
template<> struct DefaultHash<WebKit::TileForGrid> : TileForGridHash { };

template<> struct HashTraits<WebKit::TileRenderData> : SimpleClassHashTraits<WebKit::TileRenderData> {
    static constexpr bool emptyValueIsZero = false;
    static constexpr bool hasIsEmptyValueFunction = true;
    static WebKit::TileRenderData emptyValue() { return { HashTraits<WebKit::PDFTileRenderIdentifier>::emptyValue(), { } }; }
    static bool isEmptyValue(const WebKit::TileRenderData& data) { return HashTraits<WebKit::PDFTileRenderIdentifier>::isEmptyValue(data.renderIdentifier); }
};

} // namespace WTF

namespace WebKit {

class PDFPresentationController;

class AsyncPDFRenderer final : public WebCore::TiledBackingClient,
    public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<AsyncPDFRenderer> {
    WTF_MAKE_TZONE_ALLOCATED(AsyncPDFRenderer);
    WTF_MAKE_NONCOPYABLE(AsyncPDFRenderer);
public:
    static Ref<AsyncPDFRenderer> create(PDFPresentationController&);

    virtual ~AsyncPDFRenderer();

    void startTrackingLayer(WebCore::GraphicsLayer&);
    void stopTrackingLayer(WebCore::GraphicsLayer&);
    void teardown();

    void releaseMemory();

    bool paintTilesForPage(const WebCore::GraphicsLayer*, WebCore::GraphicsContext&, float documentScale, const WebCore::FloatRect& clipRect, const WebCore::FloatRect& clipRectInPageCoordinates, const WebCore::FloatRect& pageBoundsInPaintingCoordinates, PDFDocumentLayout::PageIndex);
    void paintPagePreview(WebCore::GraphicsContext&, const WebCore::FloatRect& clipRect, const WebCore::FloatRect& pageBoundsInPaintingCoordinates, PDFDocumentLayout::PageIndex);

    // Throws away existing tiles. Can result in flashing.
    void invalidateTilesForPaintingRect(float pageScaleFactor, const WebCore::FloatRect& paintingRect);

    // Updates existing tiles. Can result in temporarily stale content.
    void setNeedsRenderForRect(WebCore::GraphicsLayer&, const WebCore::FloatRect& bounds);
    void setNeedsPagePreviewRenderForPageCoverage(const PDFPageCoverage&);

    void generatePreviewImageForPage(PDFDocumentLayout::PageIndex, float scale);
    void removePreviewForPage(PDFDocumentLayout::PageIndex);

    void setShowDebugBorders(bool);

private:
    AsyncPDFRenderer(PDFPresentationController&);

    RefPtr<WebCore::GraphicsLayer> layerForTileGrid(WebCore::TileGridIdentifier) const;

    TileRenderInfo renderInfoForFullTile(const WebCore::TiledBacking&, const TileForGrid& tileInfo, const WebCore::FloatRect& tileRect) const;
    TileRenderInfo renderInfoForTile(const WebCore::TiledBacking&, const TileForGrid& tileInfo, const WebCore::FloatRect& tileRect, const WebCore::FloatRect& renderRect, RefPtr<WebCore::NativeImage>&& background) const;

    bool renderInfoIsValidForTile(WebCore::TiledBacking&, const TileForGrid&, const TileRenderInfo&) const;

    // TiledBackingClient
    void willRepaintTile(WebCore::TiledBacking&, WebCore::TileGridIdentifier, WebCore::TileIndex, const WebCore::FloatRect& tileRect, const WebCore::FloatRect& tileDirtyRect) final;
    void willRemoveTile(WebCore::TiledBacking&, WebCore::TileGridIdentifier, WebCore::TileIndex) final;
    void willRepaintAllTiles(WebCore::TiledBacking&, WebCore::TileGridIdentifier) final;

    void coverageRectDidChange(WebCore::TiledBacking&, const WebCore::FloatRect&) final;

    void willRevalidateTiles(WebCore::TiledBacking&, WebCore::TileGridIdentifier, WebCore::TileRevalidationType) final;
    void didRevalidateTiles(WebCore::TiledBacking&, WebCore::TileGridIdentifier, WebCore::TileRevalidationType, const UncheckedKeyHashSet<WebCore::TileIndex>& tilesNeedingDisplay) final;

    void willRepaintTilesAfterScaleFactorChange(WebCore::TiledBacking&, WebCore::TileGridIdentifier) final;
    void didRepaintTilesAfterScaleFactorChange(WebCore::TiledBacking&, WebCore::TileGridIdentifier) final;

    void didAddGrid(WebCore::TiledBacking&, WebCore::TileGridIdentifier) final;
    void willRemoveGrid(WebCore::TiledBacking&, WebCore::TileGridIdentifier) final;

    std::optional<PDFTileRenderIdentifier> enqueueTileRenderForTileGridRepaint(WebCore::TiledBacking&, WebCore::TileGridIdentifier, WebCore::TileIndex, const WebCore::FloatRect& tileRect, const WebCore::FloatRect& tileDirtyRect);
    std::optional<PDFTileRenderIdentifier> enqueueTileRenderIfNecessary(const TileForGrid&, TileRenderInfo&&);

    void serviceRequestQueue();

    void didCompleteTileRender(RefPtr<WebCore::NativeImage>&&, const TileForGrid&, const TileRenderData&);

    struct RevalidationStateForGrid {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        bool inFullTileRevalidation { false };
        bool inScaleChangeRepaint { false };
        HashSet<PDFTileRenderIdentifier> renderIdentifiersForCurrentRevalidation;
    };

    RevalidationStateForGrid& revalidationStateForGrid(WebCore::TileGridIdentifier);
    void trackRendersForStaleTileMaintenance(WebCore::TileGridIdentifier, HashSet<PDFTileRenderIdentifier>&&);
    void trackRenderCompletionForStaleTileMaintenance(WebCore::TileGridIdentifier, PDFTileRenderIdentifier);

    void clearRequestsAndCachedTiles();

    void didCompletePagePreviewRender(RefPtr<WebCore::NativeImage>&&, const PagePreviewRequest&);
    void removePagePreviewsOutsideCoverageRect(const WebCore::FloatRect&, const std::optional<PDFLayoutRow>& = { });

    Ref<ConcurrentWorkQueue> protectedPaintingWorkQueue() { return m_paintingWorkQueue; }

    static WebCore::FloatRect convertTileRectToPaintingCoords(const WebCore::FloatRect&, float pageScaleFactor);
    static WebCore::AffineTransform tileToPaintingTransform(float tilingScaleFactor);
    static WebCore::AffineTransform paintingToTileTransform(float tilingScaleFactor);

    ThreadSafeWeakPtr<PDFPresentationController> m_presentationController;

    HashMap<WebCore::PlatformLayerIdentifier, Ref<WebCore::GraphicsLayer>> m_layerIDtoLayerMap;
    HashMap<WebCore::TileGridIdentifier, WebCore::PlatformLayerIdentifier> m_tileGridToLayerIDMap;

    Ref<ConcurrentWorkQueue> m_paintingWorkQueue;

    HashMap<TileForGrid, TileRenderData> m_currentValidTileRenders;

    const int m_maxConcurrentTileRenders { 4 };
    int m_numConcurrentTileRenders { 0 };
    ListHashSet<TileForGrid> m_requestWorkQueue;

    struct RenderedTile {
        RefPtr<WebCore::NativeImage> image;
        TileRenderInfo tileInfo;
    };
    HashMap<TileForGrid, RenderedTile> m_rendereredTiles;
    HashMap<TileForGrid, RenderedTile> m_rendereredTilesForOldState;

    HashMap<WebCore::TileGridIdentifier, std::unique_ptr<RevalidationStateForGrid>> m_gridRevalidationState;

    struct RenderedPagePreview {
        RefPtr<WebCore::NativeImage> image;
        float scale { 1.0f };
    };
    using PDFPageIndexSet = HashSet<PDFDocumentLayout::PageIndex, IntHash<PDFDocumentLayout::PageIndex>, WTF::UnsignedWithZeroKeyHashTraits<PDFDocumentLayout::PageIndex>>;
    using PDFPageIndexToPreviewHash = HashMap<PDFDocumentLayout::PageIndex, PagePreviewRequest, IntHash<PDFDocumentLayout::PageIndex>, WTF::UnsignedWithZeroKeyHashTraits<PDFDocumentLayout::PageIndex>>;
    using PDFPageIndexToBufferHash = HashMap<PDFDocumentLayout::PageIndex, RenderedPagePreview, IntHash<PDFDocumentLayout::PageIndex>, WTF::UnsignedWithZeroKeyHashTraits<PDFDocumentLayout::PageIndex>>;

    PDFPageIndexToPreviewHash m_enqueuedPagePreviews;
    PDFPageIndexToBufferHash m_pagePreviews;

    bool m_showDebugBorders { false };
};


} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF)
