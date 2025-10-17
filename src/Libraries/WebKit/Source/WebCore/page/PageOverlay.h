/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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
#include "FloatPoint.h"
#include "IntRect.h"
#include "Timer.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WallTime.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GraphicsContext;
class GraphicsLayer;
class LocalFrame;
class Page;
class PageOverlay;
class PageOverlayController;
class PlatformMouseEvent;

class PageOverlayClient {
protected:
    virtual ~PageOverlayClient() = default;

public:
    virtual void willMoveToPage(PageOverlay&, Page*) = 0;
    virtual void didMoveToPage(PageOverlay&, Page*) = 0;
    virtual void drawRect(PageOverlay&, GraphicsContext&, const IntRect& dirtyRect) = 0;
    virtual bool mouseEvent(PageOverlay&, const PlatformMouseEvent&) = 0;
    virtual void didScrollFrame(PageOverlay&, LocalFrame&) { }

    virtual bool copyAccessibilityAttributeStringValueForPoint(PageOverlay&, String /* attribute */, FloatPoint, String&) { return false; }
    virtual bool copyAccessibilityAttributeBoolValueForPoint(PageOverlay&, String /* attribute */, FloatPoint, bool&)  { return false; }
    virtual Vector<String> copyAccessibilityAttributeNames(PageOverlay&, bool /* parameterizedNames */)  { return { }; }
};

class PageOverlay final : public RefCountedAndCanMakeWeakPtr<PageOverlay> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PageOverlay, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(PageOverlay);
public:
    enum class OverlayType : bool {
        View, // Fixed to the view size; does not scale or scroll with the document, repaints on scroll.
        Document, // Scales and scrolls with the document.
    };

    enum class AlwaysTileOverlayLayer : bool {
        Yes,
        No,
    };

    WEBCORE_EXPORT static Ref<PageOverlay> create(PageOverlayClient&, OverlayType = OverlayType::View, AlwaysTileOverlayLayer = AlwaysTileOverlayLayer::No);
    WEBCORE_EXPORT virtual ~PageOverlay();

    WEBCORE_EXPORT PageOverlayController* controller() const;

    typedef uint64_t PageOverlayID;
    virtual PageOverlayID pageOverlayID() const { return m_pageOverlayID; }

    void setPage(Page*);
    WEBCORE_EXPORT Page* page() const;
    WEBCORE_EXPORT void setNeedsDisplay(const IntRect& dirtyRect);
    WEBCORE_EXPORT void setNeedsDisplay();

    void drawRect(GraphicsContext&, const IntRect& dirtyRect);
    bool mouseEvent(const PlatformMouseEvent&);
    void didScrollFrame(LocalFrame&);

    bool copyAccessibilityAttributeStringValueForPoint(String attribute, FloatPoint parameter, String& value);
    bool copyAccessibilityAttributeBoolValueForPoint(String attribute, FloatPoint parameter, bool& value);
    Vector<String> copyAccessibilityAttributeNames(bool parameterizedNames);
    
    void startFadeInAnimation();
    void startFadeOutAnimation();
    WEBCORE_EXPORT void stopFadeOutAnimation();

    WEBCORE_EXPORT void clear();

    PageOverlayClient& client() const { return m_client; }

    enum class FadeMode : bool { DoNotFade, Fade };

    OverlayType overlayType() { return m_overlayType; }
    AlwaysTileOverlayLayer alwaysTileOverlayLayer() { return m_alwaysTileOverlayLayer; }

    WEBCORE_EXPORT IntRect bounds() const;
    WEBCORE_EXPORT IntRect frame() const;
    WEBCORE_EXPORT void setFrame(IntRect);

    WEBCORE_EXPORT IntSize viewToOverlayOffset() const;

    const Color& backgroundColor() const { return m_backgroundColor; }
    void setBackgroundColor(const Color&);

    void setShouldIgnoreMouseEventsOutsideBounds(bool flag) { m_shouldIgnoreMouseEventsOutsideBounds = flag; }

    // FIXME: PageOverlay should own its layer, instead of PageOverlayController.
    WEBCORE_EXPORT GraphicsLayer& layer();

    bool needsSynchronousScrolling() const { return m_needsSynchronousScrolling; }
    void setNeedsSynchronousScrolling(bool needsSynchronousScrolling) { m_needsSynchronousScrolling = needsSynchronousScrolling; }

private:
    explicit PageOverlay(PageOverlayClient&, OverlayType, AlwaysTileOverlayLayer);

    void startFadeAnimation();
    void fadeAnimationTimerFired();

    PageOverlayClient& m_client;
    WeakPtr<Page> m_page;

    Timer m_fadeAnimationTimer;
    WallTime m_fadeAnimationStartTime;
    Seconds m_fadeAnimationDuration;

    enum FadeAnimationType {
        NoAnimation,
        FadeInAnimation,
        FadeOutAnimation,
    };

    FadeAnimationType m_fadeAnimationType { NoAnimation };
    float m_fractionFadedIn { 1 };

    bool m_needsSynchronousScrolling;

    OverlayType m_overlayType;
    AlwaysTileOverlayLayer m_alwaysTileOverlayLayer;
    IntRect m_overrideFrame;

    Color m_backgroundColor { Color::transparentBlack };
    PageOverlayID m_pageOverlayID;

    bool m_shouldIgnoreMouseEventsOutsideBounds { true };
};

} // namespace WebKit
