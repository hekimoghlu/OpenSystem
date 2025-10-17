/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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

#if ENABLE(UNIFIED_PDF_DATA_DETECTION)

#include "PDFDocumentLayout.h"
#include <WebCore/DataDetectorHighlight.h>
#include <WebCore/GraphicsLayer.h>
#include <WebCore/PageOverlay.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {
class GraphicsContext;
class GraphicsLayer;
class GraphicsLayerClient;
class IntRect;
class LocalFrame;
class PlatformMouseEvent;

enum class RenderingUpdateStep : uint32_t;
}

namespace WebKit {

class PDFDataDetectorItem;
class UnifiedPDFPlugin;
class WebMouseEvent;

class PDFDataDetectorOverlayController final : private WebCore::PageOverlayClient, WebCore::DataDetectorHighlightClient {
    WTF_MAKE_TZONE_ALLOCATED(PDFDataDetectorOverlayController);
    WTF_MAKE_NONCOPYABLE(PDFDataDetectorOverlayController);
public:
    explicit PDFDataDetectorOverlayController(UnifiedPDFPlugin&);
    virtual ~PDFDataDetectorOverlayController();
    void teardown();

    bool handleMouseEvent(const WebMouseEvent&, PDFDocumentLayout::PageIndex);
    RefPtr<WebCore::PageOverlay> protectedOverlay() const { return m_overlay; }

    enum class ShouldUpdatePlatformHighlightData : bool { No, Yes };
    enum class ActiveHighlightChanged : bool { No, Yes };
    void didInvalidateHighlightOverlayRects(std::optional<PDFDocumentLayout::PageIndex> = { }, ShouldUpdatePlatformHighlightData = ShouldUpdatePlatformHighlightData::Yes, ActiveHighlightChanged = ActiveHighlightChanged::No);
    void hideActiveHighlightOverlay();

private:
    // PageOverlayClient
    void willMoveToPage(WebCore::PageOverlay&, WebCore::Page*) final;
    void didMoveToPage(WebCore::PageOverlay&, WebCore::Page*) final { }
    void drawRect(WebCore::PageOverlay&, WebCore::GraphicsContext&, const WebCore::IntRect&) final { }
    bool mouseEvent(WebCore::PageOverlay&, const WebCore::PlatformMouseEvent&) final { return false; }
    void didScrollFrame(WebCore::PageOverlay&, WebCore::LocalFrame&) final { }

    // DataDetectorHighlightClient
    WebCore::DataDetectorHighlight* activeHighlight() const final { return m_activeDataDetectorItemWithHighlight.second.get(); }
    void scheduleRenderingUpdate(OptionSet<WebCore::RenderingUpdateStep>) final;
    float deviceScaleFactor() const final;
    RefPtr<WebCore::GraphicsLayer> createGraphicsLayer(WebCore::GraphicsLayerClient&) final;

    WebCore::PageOverlay& installOverlayIfNeeded();
    void uninstallOverlay();

    RetainPtr<DDHighlightRef> createPlatformDataDetectorHighlight(PDFDataDetectorItem&) const;
    void updatePlatformHighlightData(PDFDocumentLayout::PageIndex);
    void updateDataDetectorHighlightsIfNeeded(PDFDocumentLayout::PageIndex);

    bool handleDataDetectorAction(const WebCore::IntPoint&, PDFDataDetectorItem&);

    RefPtr<UnifiedPDFPlugin> protectedPlugin() const;

    ThreadSafeWeakPtr<UnifiedPDFPlugin> m_plugin;

    RefPtr<WebCore::PageOverlay> m_overlay;

    using PDFDataDetectorItemWithHighlight = std::pair<Ref<PDFDataDetectorItem>, Ref<WebCore::DataDetectorHighlight>>;
    using PDFDataDetectorItemWithHighlightPtr = std::pair<RefPtr<PDFDataDetectorItem>, RefPtr<WebCore::DataDetectorHighlight>>;
    using PDFDataDetectorItemsWithHighlights = Vector<PDFDataDetectorItemWithHighlight>;
    template <typename Key, typename Value>
    using HashMapWithUnsignedIntegralZeroKeyAllowed = HashMap<Key, Value, WTF::IntHash<Key>, WTF::UnsignedWithZeroKeyHashTraits<Key>>;
    using PDFDataDetectorItemsWithHighlightsMap = HashMapWithUnsignedIntegralZeroKeyAllowed<PDFDocumentLayout::PageIndex, PDFDataDetectorItemsWithHighlights>;

    PDFDataDetectorItemsWithHighlightsMap m_pdfDataDetectorItemsWithHighlightsMap;
    PDFDataDetectorItemWithHighlightPtr m_activeDataDetectorItemWithHighlight;
    PDFDataDetectorItemWithHighlightPtr m_staleDataDetectorItemWithHighlight;
};

} // namespace WebKit

#endif // ENABLE(UNIFIED_PDF_DATA_DETECTION)
