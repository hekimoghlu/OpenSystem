/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

#include "RenderPtr.h"
#include "RenderStyleConstants.h"
#include "Scrollbar.h"
#include <wtf/HashMap.h>

namespace WebCore {

class Element;
class LocalFrame;
class RenderBox;
class RenderScrollbarPart;
class RenderStyle;

class RenderScrollbar final : public Scrollbar {
public:
    friend class Scrollbar;
    static Ref<Scrollbar> createCustomScrollbar(ScrollableArea&, ScrollbarOrientation, Element*, LocalFrame* owningFrame = nullptr);
    virtual ~RenderScrollbar();

    RenderBox* owningRenderer() const;

    void paintPart(GraphicsContext&, ScrollbarPart, const IntRect&);

    IntRect buttonRect(ScrollbarPart) const;
    IntRect trackRect(int startLength, int endLength) const;
    IntRect trackPieceRectWithMargins(ScrollbarPart, const IntRect&) const;

    int minimumThumbLength() const;

    float opacity() const;
    
    bool isHiddenByStyle() const override;

    std::unique_ptr<RenderStyle> getScrollbarPseudoStyle(ScrollbarPart, PseudoId) const;

private:
    RenderScrollbar(ScrollableArea&, ScrollbarOrientation, Element*, LocalFrame*);

    bool isOverlayScrollbar() const override { return false; }

    void setParent(ScrollView*) override;
    void setEnabled(bool) override;

    void paint(GraphicsContext&, const IntRect& damageRect, Widget::SecurityOriginPaintPolicy, RegionContext*) override;

    void setHoveredPart(ScrollbarPart) override;
    void setPressedPart(ScrollbarPart) override;

    void styleChanged() override;

    void updateScrollbarParts();

    void updateScrollbarPart(ScrollbarPart);

    // This Scrollbar(Widget) may outlive the DOM which created it (during tear down),
    // so we keep a reference to the Element which caused this custom scrollbar creation.
    // This will not create a reference cycle as the Widget tree is owned by our containing
    // FrameView which this Element pointer can in no way keep alive. See webkit bug 80610.
    RefPtr<Element> m_ownerElement;

    WeakPtr<LocalFrame> m_owningFrame;
    UncheckedKeyHashMap<unsigned, RenderPtr<RenderScrollbarPart>> m_parts;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::RenderScrollbar)
    static bool isType(const WebCore::Scrollbar& scrollbar) { return scrollbar.isCustomScrollbar(); }
SPECIALIZE_TYPE_TRAITS_END()
