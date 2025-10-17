/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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

#include "ContentsFormat.h"
#include "PlatformCALayer.h"

OBJC_CLASS NSObject;

namespace WebCore {

class PlatformCALayerCocoa final : public PlatformCALayer {
public:
    static Ref<PlatformCALayerCocoa> create(LayerType, PlatformCALayerClient*);
    
    // This function passes the layer as a void* rather than a PlatformLayer because PlatformLayer
    // is defined differently for Obj C and C++. This allows callers from both languages.
    static Ref<PlatformCALayerCocoa> create(void* platformLayer, PlatformCALayerClient*);

    WEBCORE_EXPORT static LayerType layerTypeForPlatformLayer(PlatformLayer*);

    ~PlatformCALayerCocoa();

    void setOwner(PlatformCALayerClient*) override;

    void setNeedsDisplay() override;
    void setNeedsDisplayInRect(const FloatRect& dirtyRect) override;
    bool needsDisplay() const override;

    void copyContentsFromLayer(PlatformCALayer*) override;

    PlatformCALayer* superlayer() const override;
    void removeFromSuperlayer() override;
    void setSublayers(const PlatformCALayerList&) override;
    PlatformCALayerList sublayersForLogging() const override;
    void removeAllSublayers() override;
    void appendSublayer(PlatformCALayer&) override;
    void insertSublayer(PlatformCALayer&, size_t index) override;
    void replaceSublayer(PlatformCALayer& reference, PlatformCALayer&) override;
    const PlatformCALayerList* customSublayers() const override { return m_customSublayers.get(); }
    void adoptSublayers(PlatformCALayer& source) override;

    void addAnimationForKey(const String& key, PlatformCAAnimation&) override;
    void removeAnimationForKey(const String& key) override;
    RefPtr<PlatformCAAnimation> animationForKey(const String& key) override;
    void animationStarted(const String& key, MonotonicTime beginTime) override;
    void animationEnded(const String& key) override;

    void setMaskLayer(RefPtr<WebCore::PlatformCALayer>&&) override;

    bool isOpaque() const override;
    void setOpaque(bool) override;

    FloatRect bounds() const override;
    void setBounds(const FloatRect&) override;

    FloatPoint3D position() const override;
    void setPosition(const FloatPoint3D&) override;

    FloatPoint3D anchorPoint() const override;
    void setAnchorPoint(const FloatPoint3D&) override;

    TransformationMatrix transform() const override;
    void setTransform(const TransformationMatrix&) override;

    TransformationMatrix sublayerTransform() const override;
    void setSublayerTransform(const TransformationMatrix&) override;

    void setIsBackdropRoot(bool) final;
    bool backdropRootIsOpaque() const final;
    void setBackdropRootIsOpaque(bool) final;

    bool isHidden() const override;
    void setHidden(bool) override;

    bool contentsHidden() const override;
    void setContentsHidden(bool) override;

    bool userInteractionEnabled() const override;
    void setUserInteractionEnabled(bool) override;

    void setBackingStoreAttached(bool) override;
    bool backingStoreAttached() const override;

#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION) || HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
    void setVisibleRect(const FloatRect&) override;
#endif

    bool geometryFlipped() const override;
    WEBCORE_EXPORT void setGeometryFlipped(bool) override;

    bool isDoubleSided() const override;
    void setDoubleSided(bool) override;

    bool masksToBounds() const override;
    void setMasksToBounds(bool) override;

    bool acceleratesDrawing() const override;
    void setAcceleratesDrawing(bool) override;

    ContentsFormat contentsFormat() const override;
    void setContentsFormat(ContentsFormat) override;

    bool hasContents() const override;
    CFTypeRef contents() const override;
    void setContents(CFTypeRef) override;
    void clearContents() override;
    void setDelegatedContents(const PlatformCALayerInProcessDelegatedContents&) override;

    void setContentsRect(const FloatRect&) override;

    void setMinificationFilter(FilterType) override;
    void setMagnificationFilter(FilterType) override;

    Color backgroundColor() const override;
    void setBackgroundColor(const Color&) override;

    void setBorderWidth(float) override;

    void setBorderColor(const Color&) override;

    float opacity() const override;
    void setOpacity(float) override;
    void setFilters(const FilterOperations&) override;
    WEBCORE_EXPORT static bool filtersCanBeComposited(const FilterOperations&);
    void copyFiltersFrom(const PlatformCALayer&) override;

    void setBlendMode(BlendMode) override;

    void setName(const String&) override;

    void setSpeed(float) override;

    void setTimeOffset(CFTimeInterval) override;

    float contentsScale() const override;
    void setContentsScale(float) override;

    float cornerRadius() const override;
    void setCornerRadius(float) override;

    void setAntialiasesEdges(bool) override;

    MediaPlayerVideoGravity videoGravity() const override;
    void setVideoGravity(MediaPlayerVideoGravity) override;

    FloatRoundedRect shapeRoundedRect() const override;
    void setShapeRoundedRect(const FloatRoundedRect&) override;

    Path shapePath() const override;
    void setShapePath(const Path&) override;

    WindRule shapeWindRule() const override;
    void setShapeWindRule(WindRule) override;

    GraphicsLayer::CustomAppearance customAppearance() const override { return m_customAppearance; }
    void updateCustomAppearance(GraphicsLayer::CustomAppearance) override;

    const EventRegion* eventRegion() const override { return &m_eventRegion; }
    void setEventRegion(const EventRegion&) override;

#if ENABLE(SCROLLING_THREAD)
    std::optional<ScrollingNodeID> scrollingNodeID() const override { return m_scrollingNodeID; }
    void setScrollingNodeID(std::optional<ScrollingNodeID> nodeID) override { m_scrollingNodeID = nodeID; }
#endif

#if HAVE(CORE_ANIMATION_SEPARATED_LAYERS)
    bool isSeparated() const override;
    void setIsSeparated(bool) override;

#if HAVE(CORE_ANIMATION_SEPARATED_PORTALS)
    bool isSeparatedPortal() const override;
    void setIsSeparatedPortal(bool) override;

    bool isDescendentOfSeparatedPortal() const override;
    void setIsDescendentOfSeparatedPortal(bool) override;
#endif
#endif

#if HAVE(CORE_MATERIAL)
    AppleVisualEffect appleVisualEffect() const override;
    void setAppleVisualEffect(AppleVisualEffect) override;
#endif

    TiledBacking* tiledBacking() override;

    Ref<PlatformCALayer> clone(PlatformCALayerClient* owner) const override;

    Ref<PlatformCALayer> createCompatibleLayer(PlatformCALayer::LayerType, PlatformCALayerClient*) const override;

    void enumerateRectsBeingDrawn(GraphicsContext&, void (^block)(FloatRect)) override;

    unsigned backingStoreBytesPerPixel() const override;

private:
    PlatformCALayerCocoa(LayerType, PlatformCALayerClient* owner);
    PlatformCALayerCocoa(PlatformLayer*, PlatformCALayerClient* owner);

    void commonInit();

    Type type() const final { return Type::Cocoa; }

    bool requiresCustomAppearanceUpdateOnBoundsChange() const;

    void updateContentsFormat();

    AVPlayerLayer *avPlayerLayer() const;

    RetainPtr<NSObject> m_delegate;
    std::unique_ptr<PlatformCALayerList> m_customSublayers;
    GraphicsLayer::CustomAppearance m_customAppearance { GraphicsLayer::CustomAppearance::None };
#if HAVE(CORE_MATERIAL)
    AppleVisualEffect m_appleVisualEffect { AppleVisualEffect::None };
#endif
    std::unique_ptr<FloatRoundedRect> m_shapeRoundedRect;
#if ENABLE(SCROLLING_THREAD)
    Markable<ScrollingNodeID> m_scrollingNodeID;
#endif
    EventRegion m_eventRegion;
    ContentsFormat m_contentsFormat { ContentsFormat::RGBA8 };
    bool m_backingStoreAttached { true };
    bool m_backdropRootIsOpaque { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_PLATFORM_CALAYER(WebCore::PlatformCALayerCocoa, type() == WebCore::PlatformCALayer::Type::Cocoa)
