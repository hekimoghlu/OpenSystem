/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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

#include "WCUpdateInfo.h"
#include <WebCore/GraphicsLayerContentsDisplayDelegate.h>
#include <wtf/DoublyLinkedList.h>

namespace WebCore {
class TransformState;
}

namespace WebKit {
class WCTiledBacking;

class GraphicsLayerWC final : public WebCore::GraphicsLayer, public DoublyLinkedListNode<GraphicsLayerWC> {
public:
    struct Observer {
        virtual void graphicsLayerAdded(GraphicsLayerWC&) = 0;
        virtual void graphicsLayerRemoved(GraphicsLayerWC&) = 0;
        virtual void commitLayerUpdateInfo(WCLayerUpdateInfo&&) = 0;
        virtual RefPtr<WebCore::ImageBuffer> createImageBuffer(WebCore::FloatSize, float deviceScaleFactor) = 0;
    };

    GraphicsLayerWC(Type layerType, WebCore::GraphicsLayerClient&, Observer&);
    ~GraphicsLayerWC() override;

    void clearObserver() { m_observer = nullptr; }

    // GraphicsLayer
    std::optional<WebCore::PlatformLayerIdentifier> primaryLayerID() const override;
    void setNeedsDisplay() override;
    void setNeedsDisplayInRect(const WebCore::FloatRect&, ShouldClipToLayer) override;
    void setContentsNeedsDisplay() override;
    bool setChildren(Vector<Ref<GraphicsLayer>>&&) override;
    void addChild(Ref<GraphicsLayer>&&) override;
    void addChildAtIndex(Ref<GraphicsLayer>&&, int index) override;
    void addChildAbove(Ref<GraphicsLayer>&&, GraphicsLayer* sibling) override;
    void addChildBelow(Ref<GraphicsLayer>&&, GraphicsLayer* sibling) override;
    bool replaceChild(GraphicsLayer* oldChild, Ref<GraphicsLayer>&& newChild) override;
    void willModifyChildren() override;
    void setMaskLayer(RefPtr<GraphicsLayer>&&) override;
    void setReplicatedLayer(GraphicsLayer*) override;
    void setReplicatedByLayer(RefPtr<GraphicsLayer>&&) override;
    void setPosition(const WebCore::FloatPoint&) override;
    void syncPosition(const WebCore::FloatPoint&) override;
    void setAnchorPoint(const WebCore::FloatPoint3D&) override;
    void setSize(const WebCore::FloatSize&) override;
    void setBoundsOrigin(const WebCore::FloatPoint&) override;
    void syncBoundsOrigin(const WebCore::FloatPoint&) override;
    void setTransform(const WebCore::TransformationMatrix&) override;
    void setChildrenTransform(const WebCore::TransformationMatrix&) override;
    void setPreserves3D(bool) override;
    void setMasksToBounds(bool) override;
    void setBackgroundColor(const WebCore::Color&) override;
    void setOpacity(float) override;
    void setContentsRect(const WebCore::FloatRect&) override;
    void setContentsClippingRect(const WebCore::FloatRoundedRect&) override;
    void setContentsRectClipsDescendants(bool) override;
    void setDrawsContent(bool) override;
    void setContentsVisible(bool) override;
    void setBackfaceVisibility(bool) override;
    void setContentsToSolidColor(const WebCore::Color&) override;
    void setContentsToPlatformLayer(PlatformLayer*, ContentsLayerPurpose) override;
    void setContentsToPlatformLayerHost(WebCore::LayerHostingContextIdentifier) override;
    void setContentsDisplayDelegate(RefPtr<WebCore::GraphicsLayerContentsDisplayDelegate>&&, ContentsLayerPurpose) override;
    bool shouldDirectlyCompositeImage(WebCore::Image*) const override { return false; }
    bool usesContentsLayer() const override;
    void setShowDebugBorder(bool) override;
    void setDebugBorder(const WebCore::Color&, float width) override;
    void setShowRepaintCounter(bool) override;
    bool setFilters(const WebCore::FilterOperations&) override;
    bool setBackdropFilters(const WebCore::FilterOperations&) override;
    void setBackdropFiltersRect(const WebCore::FloatRoundedRect&) override;
    void flushCompositingState(const WebCore::FloatRect& clipRect) override;
    void flushCompositingStateForThisLayerOnly() override;
    WebCore::TiledBacking* tiledBacking() const override;

protected:
    friend WCTiledBacking;

    RefPtr<WebCore::ImageBuffer> createImageBuffer(WebCore::FloatSize, float deviceScaleFactor);
    
private:
    struct VisibleAndCoverageRects {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        WebCore::FloatRect visibleRect;
        WebCore::FloatRect coverageRect;
        WebCore::TransformationMatrix animatingTransform;
    };

    enum ScheduleFlushOrNot { ScheduleFlush, DontScheduleFlush };
    void noteLayerPropertyChanged(OptionSet<WCLayerChange>, ScheduleFlushOrNot = ScheduleFlush);
    WebCore::TransformationMatrix transformByApplyingAnchorPoint(const WebCore::TransformationMatrix&) const;
    WebCore::TransformationMatrix layerTransform(const WebCore::FloatPoint&, const WebCore::TransformationMatrix* = nullptr) const;
    VisibleAndCoverageRects computeVisibleAndCoverageRect(WebCore::TransformState&, bool preserves3D) const;
    void recursiveCommitChanges(const WebCore::TransformState&);

    friend class WTF::DoublyLinkedListNode<GraphicsLayerWC>;

    GraphicsLayerWC* m_prev;
    GraphicsLayerWC* m_next;
    WebCore::PlatformLayerIdentifier m_layerID { WebCore::PlatformLayerIdentifier::generate() };
    Observer* m_observer;
    std::unique_ptr<WCTiledBacking> m_tiledBacking;
    PlatformLayer* m_platformLayer { nullptr };
    Markable<WebCore::LayerHostingContextIdentifier> m_hostIdentifier;
    WebCore::Color m_solidColor;
    WebCore::Color m_debugBorderColor;
    OptionSet<WCLayerChange> m_uncommittedChanges;
    float m_debugBorderWidth { 0 };
};

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
