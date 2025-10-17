/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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

#include "InlineIteratorBox.h"

namespace WebCore {

namespace InlineIterator {

class InlineBoxIterator;

class InlineBox : public Box {
public:
    InlineBox(PathVariant&&);

    const RenderBoxModelObject& renderer() const { return downcast<RenderBoxModelObject>(Box::renderer()); }

    RectEdges<bool> closedEdges() const;

    // FIXME: Remove. For intermediate porting steps only.
    const LegacyInlineFlowBox* legacyInlineBox() const { return downcast<LegacyInlineFlowBox>(Box::legacyInlineBox()); }

    InlineBoxIterator nextInlineBoxLineRightward() const;
    InlineBoxIterator nextInlineBoxLineLeftward() const;
    InlineBoxIterator iterator() const;

    LeafBoxIterator firstLeafBox() const;
    LeafBoxIterator lastLeafBox() const;
    LeafBoxIterator endLeafBox() const;
    inline bool isSplit() const;

    IteratorRange<BoxIterator> descendants() const;
};

class InlineBoxIterator : public BoxIterator {
public:
    InlineBoxIterator() = default;
    InlineBoxIterator(Box::PathVariant&&);
    InlineBoxIterator(const Box&);

    const InlineBox& operator*() const { return get(); }
    const InlineBox* operator->() const { return &get(); }

    InlineBoxIterator& traverseInlineBoxLineRightward();
    InlineBoxIterator& traverseInlineBoxLineLeftward();

private:
    const InlineBox& get() const { return downcast<InlineBox>(m_box); }
};

InlineBoxIterator lineLeftmostInlineBoxFor(const RenderInline&);
InlineBoxIterator firstRootInlineBoxFor(const RenderBlockFlow&);

InlineBoxIterator inlineBoxFor(const LegacyInlineFlowBox&);
InlineBoxIterator inlineBoxFor(const LayoutIntegration::InlineContent&, const InlineDisplay::Box&);
InlineBoxIterator inlineBoxFor(const LayoutIntegration::InlineContent&, size_t boxIndex);

inline InlineBoxIterator InlineBox::iterator() const
{
    return { *this };
}

inline bool InlineBox::isSplit() const
{
    return nextInlineBoxLineRightward() || nextInlineBoxLineLeftward();
}

}
}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::InlineIterator::InlineBox)
static bool isType(const WebCore::InlineIterator::Box& box) { return box.isInlineBox(); }
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::InlineIterator::InlineBoxIterator)
static bool isType(const WebCore::InlineIterator::BoxIterator& box) { return !box || box->isInlineBox(); }
SPECIALIZE_TYPE_TRAITS_END()
