/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 31, 2023.
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

#include "HTMLDivElement.h"
#include "RenderBlockFlow.h"
#include <wtf/Forward.h>

namespace WebCore {

class HTMLInputElement;
class TouchEvent;

class SliderThumbElement final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SliderThumbElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SliderThumbElement);
public:
    static Ref<SliderThumbElement> create(Document&);

    void setPositionFromValue();
    void dragFrom(const LayoutPoint&);
    RefPtr<HTMLInputElement> hostInput() const;
    void setPositionFromPoint(const LayoutPoint&);

#if ENABLE(IOS_TOUCH_EVENTS)
    void handleTouchEvent(TouchEvent&);
#endif

    void hostDisabledStateChanged();

private:
    explicit SliderThumbElement(Document&);
    bool isSliderThumbElement() const final { return true; }

    Ref<Element> cloneElementWithoutAttributesAndChildren(TreeScope&) final;
    bool isDisabledFormControl() const final;
    bool matchesReadWritePseudoClass() const final;

    void defaultEventHandler(Event&) final;
    bool willRespondToMouseMoveEvents() const final;
    bool willRespondToMouseClickEventsWithEditability(Editability) const final;

#if ENABLE(IOS_TOUCH_EVENTS)
    void didAttachRenderers() final;
#endif
    void willDetachRenderers() final;

    std::optional<Style::ResolvedStyle> resolveCustomStyle(const Style::ResolutionContext&, const RenderStyle*) final;

    void startDragging();
    void stopDragging();

#if ENABLE(IOS_TOUCH_EVENTS)
    unsigned exclusiveTouchIdentifier() const;
    void setExclusiveTouchIdentifier(unsigned);
    void clearExclusiveTouchIdentifier();

    void handleTouchStart(TouchEvent&);
    void handleTouchMove(TouchEvent&);
    void handleTouchEndAndCancel(TouchEvent&);

    bool shouldAcceptTouchEvents();
    void registerForTouchEvents();
    void unregisterForTouchEvents();
#endif

    bool m_inDragMode { false };

#if ENABLE(IOS_TOUCH_EVENTS)
    // FIXME: Currently it is safe to use 0, but this may need to change
    // if touch identifiers change in the future and can be 0.
    static const unsigned NoIdentifier = 0;
    unsigned m_exclusiveTouchIdentifier { NoIdentifier };
    bool m_isRegisteredAsTouchEventListener { false };
#endif
};

// --------------------------------

class SliderContainerElement final : public HTMLDivElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SliderContainerElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SliderContainerElement);
public:
    static Ref<SliderContainerElement> create(Document&);

private:
    explicit SliderContainerElement(Document&);
    RenderPtr<RenderElement> createElementRenderer(RenderStyle&&, const RenderTreePosition&) final;
    bool isSliderContainerElement() const final { return true; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SliderThumbElement)
    static bool isType(const WebCore::Element& element) { return element.isSliderThumbElement(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* element = dynamicDowncast<WebCore::Element>(node);
        return element && isType(*element);
    }
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SliderContainerElement)
    static bool isType(const WebCore::Element& element) { return element.isSliderContainerElement(); }
    static bool isType(const WebCore::Node& node)
    {
        auto* element = dynamicDowncast<WebCore::Element>(node);
        return element && isType(*element);
    }
SPECIALIZE_TYPE_TRAITS_END()
