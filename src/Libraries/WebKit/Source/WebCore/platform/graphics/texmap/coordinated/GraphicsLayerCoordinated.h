/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#include "GraphicsLayer.h"
#include "GraphicsLayerTransform.h"
#include "TextureMapperAnimation.h"
#include <wtf/OptionSet.h>

namespace WebCore {
class CoordinatedPlatformLayer;
class CoordinatedPlatformLayerBufferProxy;
class NativeImage;

class GraphicsLayerCoordinated final : public GraphicsLayer {
public:
    WEBCORE_EXPORT GraphicsLayerCoordinated(Type, GraphicsLayerClient&, Ref<CoordinatedPlatformLayer>&&);
    virtual ~GraphicsLayerCoordinated();

    CoordinatedPlatformLayer& coordinatedPlatformLayer() const { return m_platformLayer.get(); }

    static void clampToSizeIfRectIsInfinite(FloatRect&, const FloatSize&);

private:
    bool isGraphicsLayerCoordinated() const override { return true; }

    std::optional<PlatformLayerIdentifier> primaryLayerID() const override;

    void setPosition(const FloatPoint&) override;
    void syncPosition(const FloatPoint&) override;
    void setBoundsOrigin(const FloatPoint&) override;
    void syncBoundsOrigin(const FloatPoint&) override;
    void setAnchorPoint(const FloatPoint3D&) override;
    void setSize(const FloatSize&) override;

#if ENABLE(SCROLLING_THREAD)
    void setScrollingNodeID(std::optional<ScrollingNodeID>) override;
#endif

    void setTransform(const TransformationMatrix&) override;
    void setChildrenTransform(const TransformationMatrix&) override;

    void setDrawsContent(bool) override;
    void setMasksToBounds(bool) override;
    void setPreserves3D(bool) override;
    void setBackfaceVisibility(bool) override;
    void setOpacity(float) override;
    void setContentsVisible(bool) override;
    void setContentsOpaque(bool) override;
    void setContentsRect(const FloatRect&) override;
    void setContentsRectClipsDescendants(bool) override;
    void setContentsTileSize(const FloatSize&) override;
    void setContentsTilePhase(const FloatSize&) override;
    void setContentsClippingRect(const FloatRoundedRect&) override;
    void setContentsNeedsDisplay() override;
    void setContentsToPlatformLayer(PlatformLayer*, ContentsLayerPurpose) override;
    void setContentsDisplayDelegate(RefPtr<GraphicsLayerContentsDisplayDelegate>&&, ContentsLayerPurpose) override;
    RefPtr<GraphicsLayerAsyncContentsDisplayDelegate> createAsyncContentsDisplayDelegate(GraphicsLayerAsyncContentsDisplayDelegate*) override;
    void setContentsToImage(Image*) override;
    void setContentsToSolidColor(const Color&) override;
    bool usesContentsLayer() const override;

    bool setChildren(Vector<Ref<GraphicsLayer>>&&) override;
    void addChild(Ref<GraphicsLayer>&&) override;
    void addChildAtIndex(Ref<GraphicsLayer>&&, int index) override;
    void addChildAbove(Ref<GraphicsLayer>&&, GraphicsLayer*) override;
    void addChildBelow(Ref<GraphicsLayer>&&, GraphicsLayer*) override;
    bool replaceChild(GraphicsLayer* oldChild, Ref<GraphicsLayer>&& newChild) override;
    void willModifyChildren() override;

    void setEventRegion(EventRegion&&) override;

    void deviceOrPageScaleFactorChanged() override;

    float rootRelativeScaleFactor() const { return m_rootRelativeScaleFactor; }
    void setShouldUpdateRootRelativeScaleFactor(bool value) override { m_shouldUpdateRootRelativeScaleFactor = value; }
    void updateRootRelativeScale();

    bool setFilters(const FilterOperations&) override;
    void setMaskLayer(RefPtr<GraphicsLayer>&&) override;
    void setReplicatedByLayer(RefPtr<GraphicsLayer>&&) override;
    bool setBackdropFilters(const FilterOperations&) override;
    void setBackdropFiltersRect(const FloatRoundedRect&) override;

    bool addAnimation(const KeyframeValueList&, const FloatSize&, const Animation*, const String&, double) override;
    void removeAnimation(const String&, std::optional<AnimatedProperty>) override;
    void pauseAnimation(const String& animationName, double timeOffset) override;
    void suspendAnimations(MonotonicTime) override;
    void resumeAnimations() override;
    void transformRelatedPropertyDidChange() override;
    Vector<std::pair<String, double>> acceleratedAnimationsForTesting(const Settings&) const override;

    void setNeedsDisplay() override;
    void setNeedsDisplayInRect(const FloatRect&, ShouldClipToLayer = ClipToLayer) override;

    FloatSize pixelAlignmentOffset() const override { return m_pixelAlignmentOffset; }

    void flushCompositingState(const FloatRect&) override;
    void flushCompositingStateForThisLayerOnly() override;

    void setShowDebugBorder(bool) override;
    void setShowRepaintCounter(bool) override;
    void dumpAdditionalProperties(TextStream&, OptionSet<LayerTreeAsTextOptions>) const override;

    enum class Change : uint32_t {
        Geometry                     = 1 << 0,
        Transform                    = 1 << 1,
        ChildrenTransform            = 1 << 2,
        DrawsContent                 = 1 << 3,
        MasksToBounds                = 1 << 4,
        Preserves3D                  = 1 << 5,
        BackfaceVisibility           = 1 << 6,
        Opacity                      = 1 << 7,
        Children                     = 1 << 8,
        ContentsVisible              = 1 << 9,
        ContentsOpaque               = 1 << 10,
        ContentsRect                 = 1 << 11,
        ContentsRectClipsDescendants = 1 << 12,
        ContentsClippingRect         = 1 << 13,
        ContentsScale                = 1 << 14,
        ContentsTiling               = 1 << 15,
        ContentsBuffer               = 1 << 16,
        ContentsBufferNeedsDisplay   = 1 << 17,
        ContentsImage                = 1 << 18,
        ContentsColor                = 1 << 19,
        DirtyRegion                  = 1 << 20,
        EventRegion                  = 1 << 21,
        Filters                      = 1 << 22,
        Mask                         = 1 << 23,
        Replica                      = 1 << 24,
        Backdrop                     = 1 << 25,
        BackdropRect                 = 1 << 26,
        Animations                   = 1 << 27,
        TileCoverage                 = 1 << 28,
        DebugIndicators              = 1 << 29,
#if ENABLE(SCROLLING_THREAD)
        ScrollingNode                = 1 << 30
#endif
    };

    enum class ScheduleFlush : bool { No, Yes };
    void noteLayerPropertyChanged(OptionSet<Change>, ScheduleFlush);
    void setNeedsUpdateLayerTransform();
    std::pair<FloatPoint, float> computePositionRelativeToBase() const;
    void computePixelAlignmentIfNeeded(float pageScaleFactor, const FloatPoint& positionRelativeToBase, FloatPoint& adjustedPosition, FloatPoint& adjustedBoundsOrigin, FloatPoint3D& adjustedAnchorPoint, FloatSize& adjustedSize);
    void computeLayerTransformIfNeeded(bool affectedByTransformAnimation);
    IntRect transformedRect(const FloatRect&) const;
    IntRect transformedRectIncludingFuture(const FloatRect&) const;
    void updateGeometry(float pageScaleFactor, const FloatPoint&);
#if ENABLE(DAMAGE_TRACKING)
    void updateDamage();
#endif
    void updateDirtyRegion();
    void updateBackdropFilters();
    void updateBackdropFiltersRect();
    void updateAnimations();
    void updateVisibleRect(const FloatRect&);
    void updateIndicators();
    bool isRunningTransformAnimation() const;
    bool filtersCanBeComposited(const FilterOperations&) const;

    struct CommitState {
        FloatRect visibleRect;
        bool ancestorHadChanges { false };
        bool ancestorHasTransformAnimation { false };
    };
    void commitLayerChanges(CommitState&, float pageScaleFactor, const FloatPoint&, bool affectedByTransformAnimation);
    bool needsCommit(CommitState&) const;
    void recursiveCommitChanges(CommitState&, float pageScaleFactor = 1, const FloatPoint& positionRelativeToBase = FloatPoint(), bool affectedByPageScale = false);

    bool updateBackingStoresIfNeeded();
    bool updateBackingStoreIfNeeded();

    Ref<CoordinatedPlatformLayer> m_platformLayer;
    OptionSet<Change> m_pendingChanges;
    bool m_hasDescendantsWithPendingChanges { false };
    bool m_hasDescendantsWithPendingTilesCreation { false };
    bool m_hasDescendantsWithRunningTransformAnimations { false };
    FloatSize m_pixelAlignmentOffset;
    struct {
        bool fullRepaint { false };
        Vector<FloatRect> rects;
    } m_dirtyRegion;
    FloatRect m_visibleRect;
    struct {
        GraphicsLayerTransform current;
        GraphicsLayerTransform future;
        TransformationMatrix cachedInverse;
        TransformationMatrix cachedFutureInverse;
        TransformationMatrix cachedCombined;
    } m_layerTransform;
    bool m_needsUpdateLayerTransform { false };
    bool m_shouldUpdateRootRelativeScaleFactor : 1 { false };
    float m_rootRelativeScaleFactor { 1.0f };
    RefPtr<CoordinatedPlatformLayerBufferProxy> m_contentsBufferProxy;
    RefPtr<GraphicsLayerContentsDisplayDelegate> m_contentsDisplayDelegate;
    RefPtr<NativeImage> m_contentsImage;
    Color m_contentsColor;
    RefPtr<CoordinatedPlatformLayer> m_backdropLayer;
    TextureMapperAnimations m_animations;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_GRAPHICSLAYER(WebCore::GraphicsLayerCoordinated, isGraphicsLayerCoordinated())

#endif // USE(COORDINATED_GRAPHICS)
