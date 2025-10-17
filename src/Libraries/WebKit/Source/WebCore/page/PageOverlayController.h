/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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

#include "GraphicsLayer.h"
#include "GraphicsLayerClient.h"
#include "PageOverlay.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashMap.h>

namespace WebCore {

class LocalFrame;
class Page;
class PlatformMouseEvent;

class PageOverlayController final : public GraphicsLayerClient {
    WTF_MAKE_TZONE_ALLOCATED(PageOverlayController);
    friend class MockPageOverlayClient;
public:
    PageOverlayController(Page&);
    virtual ~PageOverlayController();

    bool hasDocumentOverlays() const;
    bool hasViewOverlays() const;

    GraphicsLayer& layerWithDocumentOverlays();
    GraphicsLayer& layerWithViewOverlays();
    Ref<GraphicsLayer> protectedLayerWithViewOverlays();

    const Vector<RefPtr<PageOverlay>>& pageOverlays() const { return m_pageOverlays; }

    WEBCORE_EXPORT void installPageOverlay(PageOverlay&, PageOverlay::FadeMode);
    WEBCORE_EXPORT void uninstallPageOverlay(PageOverlay&, PageOverlay::FadeMode);

    void setPageOverlayNeedsDisplay(PageOverlay&, const IntRect&);
    void setPageOverlayOpacity(PageOverlay&, float);
    void clearPageOverlay(PageOverlay&);
    GraphicsLayer& layerForOverlay(PageOverlay&) const;

    void didChangeViewSize();
    void didChangeDocumentSize();
    void didChangeSettings();
    WEBCORE_EXPORT void didChangeDeviceScaleFactor();
    void didChangeViewExposedRect();
    void didScrollFrame(LocalFrame&);

    void didChangeOverlayFrame(PageOverlay&);
    void didChangeOverlayBackgroundColor(PageOverlay&);

    int overlayCount() const { return m_overlayGraphicsLayers.computeSize(); }

    bool handleMouseEvent(const PlatformMouseEvent&);

    WEBCORE_EXPORT bool copyAccessibilityAttributeStringValueForPoint(String attribute, FloatPoint, String& value);
    WEBCORE_EXPORT bool copyAccessibilityAttributeBoolValueForPoint(String attribute, FloatPoint, bool& value);
    WEBCORE_EXPORT Vector<String> copyAccessibilityAttributesNames(bool parameterizedNames);

private:
    void createRootLayersIfNeeded();

    WEBCORE_EXPORT GraphicsLayer* documentOverlayRootLayer() const;
    WEBCORE_EXPORT GraphicsLayer* viewOverlayRootLayer() const;

    void installedPageOverlaysChanged();
    void attachViewOverlayLayers();
    void detachViewOverlayLayers();

    void updateSettingsForLayer(GraphicsLayer&);
    void updateForceSynchronousScrollLayerPositionUpdates();

    // GraphicsLayerClient
    void notifyFlushRequired(const GraphicsLayer*) override;
    void paintContents(const GraphicsLayer*, GraphicsContext&, const FloatRect& clipRect, OptionSet<GraphicsLayerPaintBehavior>) override;
    float deviceScaleFactor() const override;
    bool shouldSkipLayerInDump(const GraphicsLayer*, OptionSet<LayerTreeAsTextOptions>) const override;
    bool shouldDumpPropertyForLayer(const GraphicsLayer*, ASCIILiteral propertyName, OptionSet<LayerTreeAsTextOptions>) const override;
    void tiledBackingUsageChanged(const GraphicsLayer*, bool) override;

    Ref<Page> protectedPage() const;

    WeakRef<Page> m_page;
    RefPtr<GraphicsLayer> m_documentOverlayRootLayer;
    RefPtr<GraphicsLayer> m_viewOverlayRootLayer;

    WeakHashMap<PageOverlay, Ref<GraphicsLayer>> m_overlayGraphicsLayers;
    Vector<RefPtr<PageOverlay>> m_pageOverlays;
    bool m_initialized { false };
};

} // namespace WebKit
