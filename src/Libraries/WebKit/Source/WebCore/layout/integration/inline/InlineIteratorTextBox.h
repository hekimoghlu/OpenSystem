/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#include "RenderText.h"

namespace WebCore {

namespace InlineIterator {

class TextBox : public Box {
public:
    TextBox(PathVariant&&);

    bool hasHyphen() const;
    StringView originalText() const;

    unsigned start() const;
    unsigned end() const;
    unsigned length() const;

    TextBoxSelectableRange selectableRange() const;

    const FontCascade& fontCascade() const;

    inline TextRun textRun(TextRunMode = TextRunMode::Painting) const;

    const RenderText& renderer() const { return downcast<RenderText>(Box::renderer()); }

    // FIXME: Remove. For intermediate porting steps only.
    const LegacyInlineTextBox* legacyInlineBox() const { return downcast<LegacyInlineTextBox>(Box::legacyInlineBox()); }

    // This returns the next text box generated for the same RenderText/Layout::InlineTextBox.
    TextBoxIterator nextTextBox() const;
};

class TextBoxIterator : public LeafBoxIterator {
public:
    TextBoxIterator() = default;
    TextBoxIterator(Box::PathVariant&&);
    TextBoxIterator(const Box&);

    TextBoxIterator& operator++() { return traverseNextTextBox(); }

    const TextBox& operator*() const { return get(); }
    const TextBox* operator->() const { return &get(); }

    // This traverses to the next text box generated for the same RenderText/Layout::InlineTextBox.
    TextBoxIterator& traverseNextTextBox();

private:
    BoxIterator& traverseLineRightwardOnLine() = delete;
    BoxIterator& traverseLineLeftwardOnLine() = delete;
    BoxIterator& traverseLineRightwardOnLineIgnoringLineBreak() = delete;
    BoxIterator& traverseLineLeftwardOnLineIgnoringLineBreak() = delete;

    const TextBox& get() const { return downcast<TextBox>(m_box); }
};

TextBoxIterator lineLeftmostTextBoxFor(const RenderText&);
TextBoxIterator textBoxFor(const LegacyInlineTextBox*);
TextBoxIterator textBoxFor(const LayoutIntegration::InlineContent&, const InlineDisplay::Box&);
TextBoxIterator textBoxFor(const LayoutIntegration::InlineContent&, size_t boxIndex);
BoxRange<TextBoxIterator> textBoxesFor(const RenderText&);

inline bool TextBox::hasHyphen() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.hasHyphen();
    });
}

inline TextBox::TextBox(PathVariant&& path)
    : Box(WTFMove(path))
{
}

inline StringView TextBox::originalText() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.originalText();
    });
}

inline unsigned TextBox::start() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.start();
    });
}

inline unsigned TextBox::end() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.end();
    });
}

inline unsigned TextBox::length() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.length();
    });
}

inline TextBoxSelectableRange TextBox::selectableRange() const
{
    return WTF::switchOn(m_pathVariant, [&](auto& path) {
        return path.selectableRange();
    });
}

}
}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::InlineIterator::TextBox)
static bool isType(const WebCore::InlineIterator::Box& box) { return box.isText(); }
SPECIALIZE_TYPE_TRAITS_END()

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::InlineIterator::TextBoxIterator)
static bool isType(const WebCore::InlineIterator::BoxIterator& box) { return !box || box->isText(); }
SPECIALIZE_TYPE_TRAITS_END()

