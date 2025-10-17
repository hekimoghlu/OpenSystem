/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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

#include "LayoutPoint.h"
#include "UIEventWithKeyState.h"

namespace WebCore {

class LocalFrameView;

struct MouseRelatedEventInit : public EventModifierInit {
    int screenX { 0 };
    int screenY { 0 };
    double movementX { 0 };
    double movementY { 0 };
};

// Internal only: Helper class for what's common between mouse and wheel events.
class MouseRelatedEvent : public UIEventWithKeyState {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MouseRelatedEvent);
public:
    enum class IsSimulated : bool { No, Yes };

    // Note that these values are adjusted to counter the effects of zoom, so that values
    // exposed via DOM APIs are invariant under zooming.
    int screenX() const { return m_screenLocation.x(); }
    int screenY() const { return m_screenLocation.y(); }
    const IntPoint& screenLocation() const { return m_screenLocation; }
    int clientX() const { return m_clientLocation.x(); }
    int clientY() const { return m_clientLocation.y(); }
    double movementX() const { return m_movementX; }
    double movementY() const { return m_movementY; }

    const IntPoint& windowLocation() const { return m_windowLocation; }

    const LayoutPoint& clientLocation() const { return m_clientLocation; }
    int layerX() override;
    int layerY() override;
    WEBCORE_EXPORT int offsetX();
    WEBCORE_EXPORT int offsetY();
    bool isSimulated() const { return m_isSimulated; }
    void setIsSimulated(bool value) { m_isSimulated = value; }
    int pageX() const final;
    int pageY() const final;
    WEBCORE_EXPORT FloatPoint locationInRootViewCoordinates() const;

    // Page point in "absolute" coordinates (i.e. post-zoomed, page-relative coords,
    // usable with RenderObject::absoluteToLocal).
    const LayoutPoint& absoluteLocation() const { return m_absoluteLocation; }
    
    static LocalFrameView* frameViewFromWindowProxy(WindowProxy*);

    static LayoutPoint pagePointToClientPoint(LayoutPoint pagePoint, LocalFrameView*);
    static LayoutPoint pagePointToAbsolutePoint(LayoutPoint pagePoint, LocalFrameView*);

protected:
    MouseRelatedEvent(enum EventInterfaceType);
    MouseRelatedEvent();
    MouseRelatedEvent(enum EventInterfaceType, const AtomString& type, CanBubble, IsCancelable, IsComposed, MonotonicTime, RefPtr<WindowProxy>&&, int detail,
        const IntPoint& screenLocation, const IntPoint& windowLocation, double movementX, double movementY, OptionSet<Modifier> modifiers,
        IsSimulated = IsSimulated::No, IsTrusted = IsTrusted::Yes);
    MouseRelatedEvent(enum EventInterfaceType, const AtomString& type, IsCancelable, MonotonicTime, RefPtr<WindowProxy>&&, const IntPoint& globalLocation, OptionSet<Modifier>);
    MouseRelatedEvent(enum EventInterfaceType, const AtomString& type, const MouseRelatedEventInit&, IsTrusted = IsTrusted::No);

    void initCoordinates();
    void initCoordinates(const LayoutPoint& clientLocation);
    void receivedTarget() override;

    void computePageLocation();
    void computeRelativePosition();

    float documentToAbsoluteScaleFactor() const;

    // Expose these so MouseEvent::initMouseEvent can set them.
    IntPoint m_screenLocation;
    LayoutPoint m_clientLocation;

private:
    void init(bool isSimulated, const IntPoint&);

    double m_movementX { 0 };
    double m_movementY { 0 };
    LayoutPoint m_pageLocation;
    LayoutPoint m_layerLocation;
    LayoutPoint m_offsetLocation;
    LayoutPoint m_absoluteLocation;
    IntPoint m_windowLocation;
    bool m_isSimulated { false };
    bool m_hasCachedRelativePosition { false };
};

} // namespace WebCore
