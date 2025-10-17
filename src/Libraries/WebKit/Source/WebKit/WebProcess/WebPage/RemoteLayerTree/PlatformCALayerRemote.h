/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

#include "LayerProperties.h"
#include "RemoteLayerTreeContext.h"
#include "RemoteLayerTreeTransaction.h"
#include <WebCore/HTMLMediaElementIdentifier.h>
#include <WebCore/PlatformCALayer.h>
#include <WebCore/PlatformCALayerDelegatedContents.h>
#include <WebCore/PlatformLayer.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class LayerPool;
#if ENABLE(THREADED_ANIMATION_RESOLUTION)
class AcceleratedEffect;
struct AcceleratedEffectValues;
#endif
#if ENABLE(MODEL_PROCESS)
class ModelContext;
#endif
}

namespace WebKit {


using LayerHostingContextID = uint32_t;

struct PlatformCALayerRemoteDelegatedContents {
    ImageBufferBackendHandle surface;
    RefPtr<WebCore::PlatformCALayerDelegatedContentsFence> finishedFence;
    std::optional<WebCore::RenderingResourceIdentifier> surfaceIdentifier;
};

class PlatformCALayerRemote : public WebCore::PlatformCALayer, public CanMakeWeakPtr<PlatformCALayerRemote> {
public:
    static Ref<PlatformCALayerRemote> create(WebCore::PlatformCALayer::LayerType, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);
    static Ref<PlatformCALayerRemote> create(PlatformLayer *, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);
#if ENABLE(MODEL_PROCESS)
    static Ref<PlatformCALayerRemote> create(Ref<WebCore::ModelContext>, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext&);
#endif
#if ENABLE(MODEL_ELEMENT)
    static Ref<PlatformCALayerRemote> create(Ref<WebCore::Model>, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);
#endif
#if HAVE(AVKIT)
    static Ref<PlatformCALayerRemote> create(WebCore::HTMLVideoElement&, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);
#endif
    static Ref<PlatformCALayerRemote> create(const PlatformCALayerRemote&, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);

    virtual ~PlatformCALayerRemote();

    PlatformLayer* platformLayer() const override { return nullptr; }

    void recursiveBuildTransaction(RemoteLayerTreeContext&, RemoteLayerTreeTransaction&);
    void recursiveMarkWillBeDisplayedWithRenderingSuppresion();

    void setNeedsDisplayInRect(const WebCore::FloatRect& dirtyRect) override;
    void setNeedsDisplay() override;
    bool needsDisplay() const override;

    void copyContentsFromLayer(PlatformCALayer*) override;

    WebCore::PlatformCALayer* superlayer() const override;
    void removeFromSuperlayer() override;
    void setSublayers(const WebCore::PlatformCALayerList&) override;
    WebCore::PlatformCALayerList sublayersForLogging() const override { return m_children; }
    void removeAllSublayers() override;
    void appendSublayer(WebCore::PlatformCALayer&) override;
    void insertSublayer(WebCore::PlatformCALayer&, size_t index) override;
    void replaceSublayer(WebCore::PlatformCALayer& reference, WebCore::PlatformCALayer&) override;
    const WebCore::PlatformCALayerList* customSublayers() const override { return nullptr; }
    void adoptSublayers(WebCore::PlatformCALayer& source) override;

    void addAnimationForKey(const String& key, WebCore::PlatformCAAnimation&) override;
    void removeAnimationForKey(const String& key) override;
    RefPtr<WebCore::PlatformCAAnimation> animationForKey(const String& key) override;
    void animationStarted(const String& key, MonotonicTime beginTime) override;
    void animationEnded(const String& key) override;

#if ENABLE(THREADED_ANIMATION_RESOLUTION)
    void clearAcceleratedEffectsAndBaseValues() override;
    void setAcceleratedEffectsAndBaseValues(const WebCore::AcceleratedEffects&, const WebCore::AcceleratedEffectValues&) override;
#endif

    void setMaskLayer(RefPtr<WebCore::PlatformCALayer>&&) override;

    bool isOpaque() const override;
    void setOpaque(bool) override;

    WebCore::FloatRect bounds() const override;
    void setBounds(const WebCore::FloatRect&) override;

    WebCore::FloatPoint3D position() const override;
    void setPosition(const WebCore::FloatPoint3D&) override;

    WebCore::FloatPoint3D anchorPoint() const override;
    void setAnchorPoint(const WebCore::FloatPoint3D&) override;

    WebCore::TransformationMatrix transform() const override;
    void setTransform(const WebCore::TransformationMatrix&) override;

    WebCore::TransformationMatrix sublayerTransform() const override;
    void setSublayerTransform(const WebCore::TransformationMatrix&) override;

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

#if ENABLE(INTERACTION_REGIONS_IN_EVENT_REGION)
    void setVisibleRect(const WebCore::FloatRect&) override;
#endif

    bool geometryFlipped() const override;
    void setGeometryFlipped(bool) override;

    bool isDoubleSided() const override;
    void setDoubleSided(bool) override;

    bool masksToBounds() const override;
    void setMasksToBounds(bool) override;

    bool acceleratesDrawing() const override;
    void setAcceleratesDrawing(bool) override;

    WebCore::ContentsFormat contentsFormat() const override;
    void setContentsFormat(WebCore::ContentsFormat) override;

    bool hasContents() const override;
    CFTypeRef contents() const override;
    void setContents(CFTypeRef) override;
    void setDelegatedContents(const WebCore::PlatformCALayerDelegatedContents&) override;
    void setRemoteDelegatedContents(const PlatformCALayerRemoteDelegatedContents&);
    void setContentsRect(const WebCore::FloatRect&) override;

    void setMinificationFilter(WebCore::PlatformCALayer::FilterType) override;
    void setMagnificationFilter(WebCore::PlatformCALayer::FilterType) override;

    WebCore::Color backgroundColor() const override;
    void setBackgroundColor(const WebCore::Color&) override;

    void setBorderWidth(float) override;
    void setBorderColor(const WebCore::Color&) override;

    float opacity() const override;
    void setOpacity(float) override;

    void setFilters(const WebCore::FilterOperations&) override;
    static bool filtersCanBeComposited(const WebCore::FilterOperations&);
    void copyFiltersFrom(const WebCore::PlatformCALayer&) override;

    void setBlendMode(WebCore::BlendMode) override;

    void setName(const String&) override;

    void setSpeed(float) override;

    void setTimeOffset(CFTimeInterval) override;

    float contentsScale() const override;
    void setContentsScale(float) override;

    float cornerRadius() const override;
    void setCornerRadius(float) override;

    void setAntialiasesEdges(bool) override;

    WebCore::MediaPlayerVideoGravity videoGravity() const override;
    void setVideoGravity(WebCore::MediaPlayerVideoGravity) override;

    // FIXME: Having both shapeRoundedRect and shapePath is redundant. We could use shapePath for everything.
    WebCore::FloatRoundedRect shapeRoundedRect() const override;
    void setShapeRoundedRect(const WebCore::FloatRoundedRect&) override;

    WebCore::Path shapePath() const override;
    void setShapePath(const WebCore::Path&) override;

    WebCore::WindRule shapeWindRule() const override;
    void setShapeWindRule(WebCore::WindRule) override;

    WebCore::GraphicsLayer::CustomAppearance customAppearance() const override;
    void updateCustomAppearance(WebCore::GraphicsLayer::CustomAppearance) override;

    void setEventRegion(const WebCore::EventRegion&) override;

#if ENABLE(SCROLLING_THREAD)
    std::optional<WebCore::ScrollingNodeID> scrollingNodeID() const override;
    void setScrollingNodeID(std::optional<WebCore::ScrollingNodeID>) override;
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
    WebCore::AppleVisualEffect appleVisualEffect() const override;
    void setAppleVisualEffect(WebCore::AppleVisualEffect) override;
#endif

    WebCore::TiledBacking* tiledBacking() override { return nullptr; }

    Ref<WebCore::PlatformCALayer> clone(WebCore::PlatformCALayerClient* owner) const override;

    Ref<PlatformCALayer> createCompatibleLayer(WebCore::PlatformCALayer::LayerType, WebCore::PlatformCALayerClient*) const override;

    void enumerateRectsBeingDrawn(WebCore::GraphicsContext&, void (^block)(WebCore::FloatRect)) override;

    virtual uint32_t hostingContextID();

    unsigned backingStoreBytesPerPixel() const override;

    void setClonedLayer(const PlatformCALayer*);

    LayerProperties& properties() { return m_properties; }
    const LayerProperties& properties() const { return m_properties; }

    void didCommit();

    void moveToContext(RemoteLayerTreeContext&);
    RemoteLayerTreeContext* context() const { return m_context.get(); }
    
    void markFrontBufferVolatileForTesting() override;
    virtual void populateCreationProperties(RemoteLayerTreeTransaction::LayerCreationProperties&, const RemoteLayerTreeContext&, WebCore::PlatformCALayer::LayerType);

    bool containsBitmapOnly() const;

    void purgeFrontBufferForTesting() override;
    void purgeBackBufferForTesting() override;

protected:
    PlatformCALayerRemote(WebCore::PlatformCALayer::LayerType, WebCore::PlatformCALayerClient* owner, RemoteLayerTreeContext&);
    PlatformCALayerRemote(const PlatformCALayerRemote&, WebCore::PlatformCALayerClient*, RemoteLayerTreeContext&);

    void updateClonedLayerProperties(PlatformCALayerRemote& clone, bool copyContents = true) const;

private:
    Type type() const override { return Type::Remote; }
    void ensureBackingStore();
    void updateBackingStore();
    void removeSublayer(PlatformCALayerRemote*);

    WebCore::DestinationColorSpace displayColorSpace() const;

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    RemoteLayerBackingStore::IncludeDisplayList shouldIncludeDisplayListInBackingStore() const;
#endif

    bool requiresCustomAppearanceUpdateOnBoundsChange() const;

    WebCore::LayerPool* layerPool() override;

    LayerProperties m_properties;
    WebCore::PlatformCALayerList m_children;
    WeakPtr<PlatformCALayerRemote> m_superlayer;
    HashMap<String, RefPtr<WebCore::PlatformCAAnimation>> m_animations;

    bool m_acceleratesDrawing { false };
    WeakPtr<RemoteLayerTreeContext> m_context;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::PlatformCALayerRemote)
static bool isType(const WebCore::PlatformCALayer& layer)
{
    switch (layer.type()) {
    case WebCore::PlatformCALayer::Type::Cocoa:
        break;
    case WebCore::PlatformCALayer::Type::Remote:
    case WebCore::PlatformCALayer::Type::RemoteCustom:
    case WebCore::PlatformCALayer::Type::RemoteHost:
    case WebCore::PlatformCALayer::Type::RemoteModel:
        return true;
    };
    return false;
}
SPECIALIZE_TYPE_TRAITS_END()
