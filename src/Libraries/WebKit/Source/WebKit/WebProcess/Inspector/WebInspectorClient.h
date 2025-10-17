/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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

#include <WebCore/InspectorClient.h>
#include <WebCore/PageOverlay.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class GraphicsContext;
class GraphicsLayer;
class IntRect;
class PageOverlay;
}

namespace WebKit {

class WebPage;
class RepaintIndicatorLayerClient;

class WebInspectorClient : public WebCore::InspectorClient, private WebCore::PageOverlayClient {
    WTF_MAKE_TZONE_ALLOCATED(WebInspectorClient);
friend class RepaintIndicatorLayerClient;
public:
    WebInspectorClient(WebPage*);
    virtual ~WebInspectorClient();

private:
    // WebCore::InspectorClient
    void inspectedPageDestroyed() override;
    void frontendCountChanged(unsigned) override;

    Inspector::FrontendChannel* openLocalFrontend(WebCore::InspectorController*) override;
    void bringFrontendToFront() override;
    void didResizeMainFrame(WebCore::LocalFrame*) override;

    void highlight() override;
    void hideHighlight() override;

#if PLATFORM(IOS_FAMILY)
    void showInspectorIndication() override;
    void hideInspectorIndication() override;

    void didSetSearchingForNode(bool) override;
#endif

    void elementSelectionChanged(bool) override;
    void timelineRecordingChanged(bool) override;

    bool overridesShowPaintRects() const override { return true; }
    void showPaintRect(const WebCore::FloatRect&) override;
    unsigned paintRectCount() const override { return m_paintRectLayers.size(); }

    void setDeveloperPreferenceOverride(WebCore::InspectorClient::DeveloperPreference, std::optional<bool>) final;
#if ENABLE(INSPECTOR_NETWORK_THROTTLING)
    bool setEmulatedConditions(std::optional<int64_t>&& bytesPerSecondLimit) final;
#endif

    // PageOverlayClient
    void willMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override;
    void didMoveToPage(WebCore::PageOverlay&, WebCore::Page*) override;
    void drawRect(WebCore::PageOverlay&, WebCore::GraphicsContext&, const WebCore::IntRect&) override;
    bool mouseEvent(WebCore::PageOverlay&, const WebCore::PlatformMouseEvent&) override;

    void animationEndedForLayer(const WebCore::GraphicsLayer*);

    WeakPtr<WebPage> m_page;
    WeakPtr<WebCore::PageOverlay> m_highlightOverlay;
    
    RefPtr<WebCore::PageOverlay> m_paintRectOverlay;
    std::unique_ptr<RepaintIndicatorLayerClient> m_paintIndicatorLayerClient;
    HashSet<Ref<WebCore::GraphicsLayer>> m_paintRectLayers;
};

} // namespace WebKit
