/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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

#include "InlineIteratorTextBox.h"
#include "RenderSVGInlineText.h"

namespace WebCore {

class RenderSVGText;
class SVGInlineTextBox;
struct SVGTextFragment;

namespace InlineIterator {

class SVGTextBox : public TextBox {
public:
    SVGTextBox(PathVariant&&);

    FloatRect calculateBoundariesIncludingSVGTransform() const;
    LayoutRect localSelectionRect(unsigned start, unsigned end) const;
    const Vector<SVGTextFragment>& textFragments() const;

    const RenderSVGInlineText& renderer() const { return downcast<RenderSVGInlineText>(TextBox::renderer()); }

    const SVGInlineTextBox* legacyInlineBox() const;

    using Key = std::pair<const RenderSVGInlineText*, unsigned>;
};

class SVGTextBoxIterator : public TextBoxIterator {
public:
    SVGTextBoxIterator() = default;
    SVGTextBoxIterator(Box::PathVariant&&);
    SVGTextBoxIterator(const Box&);

    SVGTextBoxIterator& operator++() { return downcast<SVGTextBoxIterator>(traverseNextTextBox()); }

    const SVGTextBox& operator*() const { return get(); }
    const SVGTextBox* operator->() const { return &get(); }

private:
    const SVGTextBox& get() const { return downcast<SVGTextBox>(m_box); }
};

SVGTextBoxIterator firstSVGTextBoxFor(const RenderSVGInlineText&);
BoxRange<SVGTextBoxIterator> svgTextBoxesFor(const RenderSVGInlineText&);
SVGTextBoxIterator svgTextBoxFor(const SVGInlineTextBox*);
SVGTextBoxIterator svgTextBoxFor(const LayoutIntegration::InlineContent&, size_t boxIndex);

BoxRange<BoxIterator> boxesFor(const RenderSVGText&);

SVGTextBox::Key makeKey(const SVGTextBox&);

}
}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::InlineIterator::SVGTextBox)
static bool isType(const WebCore::InlineIterator::Box& box) { return box.isSVGText(); }
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::InlineIterator::SVGTextBoxIterator)
static bool isType(const WebCore::InlineIterator::BoxIterator& box) { return !box || box->isSVGText(); }
SPECIALIZE_TYPE_TRAITS_END()
