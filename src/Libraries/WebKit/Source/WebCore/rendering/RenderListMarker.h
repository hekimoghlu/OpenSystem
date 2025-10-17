/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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

#include "RenderBox.h"

namespace WebCore {

class CSSCounterStyle;
class RenderListItem;
class StyleRuleCounterStyle;

struct ListMarkerTextContent {
    String textWithSuffix;
    uint32_t textWithoutSuffixLength { 0 };
    TextDirection textDirection { TextDirection::LTR };
    bool isEmpty() const
    {
        return textWithSuffix.isEmpty();
    }

    StringView textWithoutSuffix() const LIFETIME_BOUND
    {
        return StringView { textWithSuffix }.left(textWithoutSuffixLength);
    }

    StringView suffix() const LIFETIME_BOUND
    {
        return StringView { textWithSuffix }.substring(textWithoutSuffixLength);
    }
};

// Used to render the list item's marker.
// The RenderListMarker always has to be a child of a RenderListItem.
class RenderListMarker final : public RenderBox {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderListMarker);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderListMarker);
public:
    RenderListMarker(RenderListItem&, RenderStyle&&);
    virtual ~RenderListMarker();

    String textWithoutSuffix() const { return m_textContent.textWithoutSuffix().toString(); };
    String textWithSuffix() const { return m_textContent.textWithSuffix; };

    bool isInside() const;

    void updateInlineMarginsAndContent();

    bool isImage() const final;

    LayoutUnit lineLogicalOffsetForListItem() const { return m_lineLogicalOffsetForListItem; }
    const RenderListItem* listItem() const;

private:
    void willBeDestroyed() final;
    ASCIILiteral renderName() const final { return "RenderListMarker"_s; }
    void computePreferredLogicalWidths() final;
    bool canHaveChildren() const final { return false; }
    void paint(PaintInfo&, const LayoutPoint&) final;
    void layout() final;
    void imageChanged(WrappedImagePtr, const IntRect*) final;
    LayoutUnit lineHeight(bool firstLine, LineDirectionMode, LinePositionMode) const final;
    LayoutUnit baselinePosition(FontBaseline, bool firstLine, LineDirectionMode, LinePositionMode) const final;
    LayoutRect selectionRectForRepaint(const RenderLayerModelObject* repaintContainer, bool clipToVisibleContent) final;
    bool canBeSelectionLeaf() const final { return true; }
    void styleWillChange(StyleDifference, const RenderStyle& newStyle) final;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;
    void computeIntrinsicLogicalWidths(LayoutUnit&, LayoutUnit&) const override { ASSERT_NOT_REACHED(); }

    void element() const = delete;

    void updateInlineMargins();
    void updateContent();
    RenderBox* parentBox(RenderBox&);
    FloatRect relativeMarkerRect();
    LayoutRect localSelectionRect();

    RefPtr<CSSCounterStyle> counterStyle() const;
    bool widthUsesMetricsOfPrimaryFont() const;

    ListMarkerTextContent m_textContent;
    RefPtr<StyleImage> m_image;

    SingleThreadWeakPtr<RenderListItem> m_listItem;
    LayoutUnit m_lineOffsetForListItem;
    LayoutUnit m_lineLogicalOffsetForListItem;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderListMarker, isRenderListMarker())
