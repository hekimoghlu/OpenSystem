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

#include "RenderBlockFlow.h"
#include "RenderListMarker.h"

namespace WebCore {

class HTMLOListElement;

class RenderListItem final : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderListItem);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderListItem);
public:
    RenderListItem(Element&, RenderStyle&&);
    virtual ~RenderListItem();

    Element& element() const { return downcast<Element>(nodeForNonAnonymous()); }

    int value() const;
    void updateValue();

    WEBCORE_EXPORT String markerTextWithoutSuffix() const;
    String markerTextWithSuffix() const;

    void updateListMarkerNumbers();

    static void updateItemValuesForOrderedList(const HTMLOListElement&);
    static unsigned itemCountForOrderedList(const HTMLOListElement&);

    RenderStyle computeMarkerStyle() const;

    RenderListMarker* markerRenderer() const { return m_marker.get(); }
    void setMarkerRenderer(RenderListMarker& marker) { m_marker = marker; }

    bool isInReversedOrderedList() const;

private:
    ASCIILiteral renderName() const final { return "RenderListItem"_s; }
    
    void paint(PaintInfo&, const LayoutPoint&) final;

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;
    void layout() final;

    void computePreferredLogicalWidths() final;

    void updateValueNow() const;
    void counterDirectivesChanged();

    SingleThreadWeakPtr<RenderListMarker> m_marker;
    mutable std::optional<int> m_value;
};

bool isHTMLListElement(const Node&);

inline int RenderListItem::value() const
{
    if (!m_value)
        updateValueNow();
    return m_value.value();
}

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderListItem, isRenderListItem())
