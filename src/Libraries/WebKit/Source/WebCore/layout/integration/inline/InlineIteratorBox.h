/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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

#include "InlineIteratorBoxLegacyPath.h"
#include "InlineIteratorBoxModernPath.h"
#include <variant>

namespace WebCore {

class RenderLineBreak;
class RenderObject;
class RenderStyle;
class RenderSVGText;

namespace InlineIterator {

class LineBoxIterator;
class BoxIterator;
class LeafBoxIterator;
class TextBoxIterator;

struct EndIterator { };

class Box {
public:
    using PathVariant = std::variant<
        BoxModernPath,
        BoxLegacyPath
    >;

    Box(PathVariant&&);

    bool isText() const;
    bool isSVGText() const;
    bool isInlineBox() const;
    bool isRootInlineBox() const;
    bool isLineBreak() const;

    FloatRect visualRect() const;
    FloatRect visualRectIgnoringBlockDirection() const;
    // Visual in inline direction, logical for writing mode.
    inline FloatRect logicalRectIgnoringInlineDirection() const;

    inline float logicalTop() const;
    inline float logicalBottom() const;
    inline float logicalHeight() const;
    inline float logicalWidth() const;
    inline float logicalLeft() const;
    inline float logicalRight() const;

    // Return line-relative left/right coords (they are still considered logical values as there's no flip for writing mode).
    inline float logicalLeftIgnoringInlineDirection() const;
    inline float logicalRightIgnoringInlineDirection() const;

    inline bool isHorizontal() const;

    unsigned minimumCaretOffset() const;
    unsigned maximumCaretOffset() const;
    unsigned leftmostCaretOffset() const { return isLeftToRightDirection() ? minimumCaretOffset() : maximumCaretOffset(); }
    unsigned rightmostCaretOffset() const { return isLeftToRightDirection() ? maximumCaretOffset() : minimumCaretOffset(); }

    // isLeftToRightDirection() here is not the same as writingMode().isBidiLTR().
    unsigned char bidiLevel() const;
    TextDirection direction() const { return bidiLevel() % 2 ? TextDirection::RTL : TextDirection::LTR; }
    bool isLeftToRightDirection() const { return direction() == TextDirection::LTR; }
    bool isInlineFlipped() const { return !(isLeftToRightDirection() == writingMode().isLogicalLeftLineLeft()); }

    RenderObject::HighlightState selectionState() const;

    const RenderObject& renderer() const;
    const RenderBlockFlow& formattingContextRoot() const;
    const RenderStyle& style() const;
    WritingMode writingMode() const { return style().writingMode(); }

    // FIXME: Remove. For intermediate porting steps only.
    const LegacyInlineBox* legacyInlineBox() const;
    const InlineDisplay::Box* inlineBox() const;

    // Text-relative left/right
    LeafBoxIterator nextLineRightwardOnLine() const;
    LeafBoxIterator nextLineLeftwardOnLine() const;
    LeafBoxIterator nextLineRightwardOnLineIgnoringLineBreak() const;
    LeafBoxIterator nextLineLeftwardOnLineIgnoringLineBreak() const;

    // Coordinate-relative left/right
    inline LeafBoxIterator nextLogicalRightwardOnLine() const;
    inline LeafBoxIterator nextLogicalLeftwardOnLine() const;
    inline LeafBoxIterator nextLogicalRightwardOnLineIgnoringLineBreak() const;
    inline LeafBoxIterator nextLogicalLeftwardOnLineIgnoringLineBreak() const;

    InlineBoxIterator parentInlineBox() const;

    LineBoxIterator lineBox() const;
    size_t lineIndex() const;

    const BoxModernPath& modernPath() const;
    const BoxLegacyPath& legacyPath() const;

protected:
    friend class BoxIterator;
    friend class InlineBoxIterator;
    friend class LeafBoxIterator;
    friend class TextBoxIterator;

    PathVariant m_pathVariant;

private:
    bool hasRenderer() const;
};

class BoxIterator {
public:
    BoxIterator() : m_box(BoxLegacyPath { nullptr }) { };
    BoxIterator(Box::PathVariant&&);
    BoxIterator(const Box&);

    explicit operator bool() const { return !atEnd(); }

    bool operator==(const BoxIterator&) const;
    bool operator==(EndIterator) const { return atEnd(); }

    const Box& operator*() const { return m_box; }
    const Box* operator->() const { return &m_box; }

    BoxIterator& traverseLineRightwardOnLine();
    BoxIterator& traverseLineRightwardOnLineSkippingChildren();

    BoxIterator& operator++() { return traverseLineRightwardOnLine(); }

    bool atEnd() const;

protected:
    Box m_box;
};

class LeafBoxIterator : public BoxIterator {
public:
    LeafBoxIterator() = default;
    LeafBoxIterator(Box::PathVariant&&);
    LeafBoxIterator(const Box&);

    // Text-relative left/right
    LeafBoxIterator& traverseLineRightwardOnLine();
    LeafBoxIterator& traverseLineLeftwardOnLine();
    LeafBoxIterator& traverseLineRightwardOnLineIgnoringLineBreak();
    LeafBoxIterator& traverseLineLeftwardOnLineIgnoringLineBreak();

    // Coordinate-relative left/right
    inline LeafBoxIterator& traverseLogicalRightwardOnLine();
    inline LeafBoxIterator& traverseLogicalLeftwardOnLine();
    inline LeafBoxIterator& traverseLogicalRightwardOnLineIgnoringLineBreak();
    inline LeafBoxIterator& traverseLogicalLeftwardOnLineIgnoringLineBreak();

    LeafBoxIterator& operator++() { return traverseLineRightwardOnLine(); }
};

template<class IteratorType>
class BoxRange {
public:
    BoxRange(IteratorType begin)
        : m_begin(begin)
    {
    }

    IteratorType begin() const { return m_begin; }
    EndIterator end() const { return { }; }

private:
    IteratorType m_begin;
};

LeafBoxIterator boxFor(const RenderLineBreak&);
LeafBoxIterator boxFor(const RenderBox&);
LeafBoxIterator boxFor(const LayoutIntegration::InlineContent&, size_t boxIndex);

// -----------------------------------------------

inline Box::Box(PathVariant&& path)
    : m_pathVariant(WTFMove(path))
{
}

inline bool Box::isText() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.isText();
    });
}

inline bool Box::isInlineBox() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.isInlineBox();
    });
}

inline bool Box::isRootInlineBox() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.isRootInlineBox();
    });
}

inline FloatRect Box::visualRectIgnoringBlockDirection() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.visualRectIgnoringBlockDirection();
    });
}

inline bool Box::isLineBreak() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.isLineBreak();
    });
}

inline unsigned Box::minimumCaretOffset() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.minimumCaretOffset();
    });
}

inline unsigned Box::maximumCaretOffset() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.maximumCaretOffset();
    });
}

inline unsigned char Box::bidiLevel() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.bidiLevel();
    });
}

inline size_t Box::lineIndex() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) {
        return path.lineIndex();
    });
}

inline const RenderObject& Box::renderer() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) -> const RenderObject& {
        return path.renderer();
    });
}

inline bool Box::hasRenderer() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) -> bool {
        return path.hasRenderer();
    });
}

inline const RenderBlockFlow& Box::formattingContextRoot() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) -> const RenderBlockFlow& {
        return path.formattingContextRoot();
    });
}

inline const RenderStyle& Box::style() const
{
    return WTF::switchOn(m_pathVariant, [](auto& path) -> const RenderStyle& {
        return path.style();
    });
}

inline const LegacyInlineBox* Box::legacyInlineBox() const
{
    if (!std::holds_alternative<BoxLegacyPath>(m_pathVariant))
        return nullptr;
    return std::get<BoxLegacyPath>(m_pathVariant).legacyInlineBox();
}

inline const InlineDisplay::Box* Box::inlineBox() const
{
    if (!std::holds_alternative<BoxModernPath>(m_pathVariant))
        return nullptr;
    return &std::get<BoxModernPath>(m_pathVariant).box();
}

}
}
