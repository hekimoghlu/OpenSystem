/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

#if !USE(COORDINATED_GRAPHICS)

#include "GraphicsLayer.h"
#include "GraphicsLayerClient.h"
#include "GraphicsLayerContentsDisplayDelegate.h"
#include "Image.h"
#include "NativeImage.h"
#include "TextureMapperAnimation.h"
#include "TextureMapperLayer.h"
#include "TextureMapperPlatformLayer.h"
#include "TextureMapperTiledBackingStore.h"

namespace WebCore {

class GraphicsLayerTextureMapper final : public GraphicsLayer, TextureMapperPlatformLayer::Client {
public:
    explicit GraphicsLayerTextureMapper(Type, GraphicsLayerClient&);
    virtual ~GraphicsLayerTextureMapper();

    // GraphicsLayer
    bool setChildren(Vector<Ref<GraphicsLayer>>&&) override;
    void addChild(Ref<GraphicsLayer>&&) override;
    void addChildAtIndex(Ref<GraphicsLayer>&&, int index) override;
    void addChildAbove(Ref<GraphicsLayer>&&, GraphicsLayer* sibling) override;
    void addChildBelow(Ref<GraphicsLayer>&&, GraphicsLayer* sibling) override;
    bool replaceChild(GraphicsLayer* oldChild, Ref<GraphicsLayer>&& newChild) override;

    void setMaskLayer(RefPtr<GraphicsLayer>&&) override;
    void setReplicatedByLayer(RefPtr<GraphicsLayer>&&) override;
    void setPosition(const FloatPoint&) override;
    void setAnchorPoint(const FloatPoint3D&) override;
    void setSize(const FloatSize&) override;
    void setTransform(const TransformationMatrix&) override;
    void setChildrenTransform(const TransformationMatrix&) override;
    void setPreserves3D(bool) override;
    void setMasksToBounds(bool) override;
    void setDrawsContent(bool) override;
    void setContentsVisible(bool) override;
    void setContentsOpaque(bool) override;
    void setBackfaceVisibility(bool) override;
    void setBackgroundColor(const Color&) override;
    void setOpacity(float) override;
    bool setFilters(const FilterOperations&) override;
    bool setBackdropFilters(const FilterOperations&) override;
    void setBackdropFiltersRect(const FloatRoundedRect&) override;

    void setNeedsDisplay() override;
    void setNeedsDisplayInRect(const FloatRect&, ShouldClipToLayer = ClipToLayer) override;
    void setContentsNeedsDisplay() override;
    void setContentsRect(const FloatRect&) override;
    void setContentsClippingRect(const FloatRoundedRect&) override;
    void setContentsRectClipsDescendants(bool) override;

    bool addAnimation(const KeyframeValueList&, const FloatSize&, const Animation*, const String&, double) override;
    void pauseAnimation(const String&, double) override;
    void removeAnimation(const String&, std::optional<AnimatedProperty>) override;

    void setContentsToImage(Image*) override;
    void setContentsToSolidColor(const Color&) override;
    void setContentsToPlatformLayer(PlatformLayer*, ContentsLayerPurpose) override;
    void setContentsDisplayDelegate(RefPtr<GraphicsLayerContentsDisplayDelegate>&&, ContentsLayerPurpose) override;
    bool usesContentsLayer() const override { return m_contentsLayer; }
    PlatformLayer* platformLayer() const override { return m_contentsLayer; }

    void setShowDebugBorder(bool) override;
    void setDebugBorder(const Color&, float width) override;
    void setShowRepaintCounter(bool) override;

    void flushCompositingState(const FloatRect&) override;
    void flushCompositingStateForThisLayerOnly() override;

    WEBCORE_EXPORT void updateBackingStoreIncludingSubLayers(TextureMapper&);

    TextureMapperLayer& layer() { return m_layer; }

    Color debugBorderColor() const { return m_debugBorderColor; }
    float debugBorderWidth() const { return m_debugBorderWidth; }

private:
    // GraphicsLayer
    bool isGraphicsLayerTextureMapper() const override { return true; }

    // TextureMapperPlatformLayer::Client
    void platformLayerWillBeDestroyed() override { setContentsToPlatformLayer(0, ContentsLayerPurpose::None); }
    void setPlatformLayerNeedsDisplay() override { setContentsNeedsDisplay(); }

    void commitLayerChanges();
    void updateDebugBorderAndRepaintCount();
    void updateBackingStoreIfNeeded(TextureMapper&);
    void prepareBackingStoreIfNeeded();
    bool shouldHaveBackingStore() const;

    bool filtersCanBeComposited(const FilterOperations&) const;

    // This set of flags help us defer which properties of the layer have been
    // modified by the compositor, so we can know what to look for in the next flush.
    enum ChangeMask {
        NoChanges =                 0,

        ChildrenChange =            (1L << 1),
        MaskLayerChange =           (1L << 2),
        ReplicaLayerChange =        (1L << 3),

        ContentChange =             (1L << 4),
        ContentsRectChange =        (1L << 5),
        ContentsVisibleChange =     (1L << 6),
        ContentsOpaqueChange =      (1L << 7),

        PositionChange =            (1L << 8),
        AnchorPointChange =         (1L << 9),
        SizeChange =                (1L << 10),
        TransformChange =           (1L << 11),
        ChildrenTransformChange =   (1L << 12),
        Preserves3DChange =         (1L << 13),

        MasksToBoundsChange =       (1L << 14),
        DrawsContentChange =        (1L << 15),
        OpacityChange =             (1L << 16),
        BackfaceVisibilityChange =  (1L << 17),

        BackingStoreChange =        (1L << 18),
        DisplayChange =             (1L << 19),
        ContentsDisplayChange =     (1L << 20),
        BackgroundColorChange =     (1L << 21),

        AnimationChange =           (1L << 22),
        FilterChange =              (1L << 23),

        DebugVisualsChange =        (1L << 24),
        RepaintCountChange =        (1L << 25),

        AnimationStarted =          (1L << 26),
        BackdropLayerChange =       (1L << 27),
        SolidColorChange =          (1L << 28),
    };
    void notifyChange(ChangeMask);

    TextureMapperLayer m_layer;
    std::unique_ptr<TextureMapperLayer> m_backdropLayer;
    RefPtr<TextureMapperTiledBackingStore> m_compositedImage;
    RefPtr<NativeImage> m_compositedNativeImage;
    RefPtr<TextureMapperTiledBackingStore> m_backingStore;

    int m_changeMask;
    bool m_needsDisplay;
    Color m_solidColor;

    Color m_debugBorderColor;
    float m_debugBorderWidth;

    TextureMapperPlatformLayer* m_contentsLayer;
    FloatRect m_needsDisplayRect;
    TextureMapperAnimations m_animations;
    MonotonicTime m_animationStartTime;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_GRAPHICSLAYER(WebCore::GraphicsLayerTextureMapper, isGraphicsLayerTextureMapper())

#endif // !USE(COORDINATED_GRAPHICS)
