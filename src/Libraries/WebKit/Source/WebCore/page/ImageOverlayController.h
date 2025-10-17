/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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

#include "Color.h"
#include "LayoutRect.h"
#include "PageOverlay.h"
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(MAC)
#include "DataDetectorHighlight.h"
#endif

namespace WebCore {

class Document;
class Element;
class GraphicsContext;
class GraphicsLayer;
class GraphicsLayerClient;
class HTMLElement;
class IntRect;
class FloatQuad;
class LocalFrame;
class Page;
class RenderElement;
class WeakPtrImplWithEventTargetData;
enum class RenderingUpdateStep : uint32_t;
struct GapRects;

class ImageOverlayController final : private PageOverlayClient
#if PLATFORM(MAC)
    , DataDetectorHighlightClient
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(ImageOverlayController);
public:
    explicit ImageOverlayController(Page&);

    void selectionQuadsDidChange(LocalFrame&, const Vector<FloatQuad>&);
    void elementUnderMouseDidChange(LocalFrame&, Element*);

#if ENABLE(DATA_DETECTION)
    WEBCORE_EXPORT bool hasActiveDataDetectorHighlightForTesting() const;
    void textRecognitionResultsChanged(HTMLElement&);
#endif

    void documentDetached(const Document&);

private:
    void willMoveToPage(PageOverlay&, Page*) final;
    void didMoveToPage(PageOverlay&, Page*) final { }
    void drawRect(PageOverlay&, GraphicsContext&, const IntRect& dirtyRect) final;
    bool mouseEvent(PageOverlay&, const PlatformMouseEvent& event) final { return platformHandleMouseEvent(event); }

    bool shouldUsePageOverlayToPaintSelection(const RenderElement&);

    PageOverlay& installPageOverlayIfNeeded();
    void uninstallPageOverlayIfNeeded();
    void uninstallPageOverlay();

#if PLATFORM(MAC)
    void updateDataDetectorHighlights(const HTMLElement&);
    void clearDataDetectorHighlights();
    bool handleDataDetectorAction(const HTMLElement&, const IntPoint&);

    // DataDetectorHighlightClient
#if ENABLE(DATA_DETECTION)
    DataDetectorHighlight* activeHighlight() const final { return m_activeDataDetectorHighlight.get(); }
    void scheduleRenderingUpdate(OptionSet<RenderingUpdateStep>) final;
    float deviceScaleFactor() const final;
    RefPtr<GraphicsLayer> createGraphicsLayer(GraphicsLayerClient&) final;
#endif
#endif

    void platformUpdateElementUnderMouse(LocalFrame&, Element* elementUnderMouse);
    bool platformHandleMouseEvent(const PlatformMouseEvent&);

    RefPtr<Page> protectedPage() const;
    RefPtr<PageOverlay> protectedOverlay() const { return m_overlay; }

    WeakPtr<Page> m_page;
    RefPtr<PageOverlay> m_overlay;
    WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData> m_hostElementForSelection;
    Vector<FloatQuad> m_selectionQuads;
    LayoutRect m_selectionClipRect;
    Color m_selectionBackgroundColor { Color::transparentBlack };

#if PLATFORM(MAC)
    using ContainerAndHighlight = std::pair<WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData>, Ref<DataDetectorHighlight>>;
    Vector<ContainerAndHighlight> m_dataDetectorContainersAndHighlights;
    RefPtr<DataDetectorHighlight> m_activeDataDetectorHighlight;
    WeakPtr<HTMLElement, WeakPtrImplWithEventTargetData> m_hostElementForDataDetectors;
#endif
};

} // namespace WebCore
