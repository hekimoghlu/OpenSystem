/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

#include "InlineIteratorSVGTextBox.h"
#include <wtf/HashMap.h>
#include <wtf/Vector.h>

namespace WebCore {

class AffineTransform;
class SVGInlineTextBox;

using SVGChunkTransformMap = UncheckedKeyHashMap<InlineIterator::SVGTextBox::Key, AffineTransform>;
using SVGTextFragmentMap = UncheckedKeyHashMap<InlineIterator::SVGTextBox::Key, Vector<SVGTextFragment>>;

// A SVGTextChunk describes a range of SVGTextFragments, see the SVG spec definition of a "text chunk".
class SVGTextChunk {
public:
    enum ChunkStyle {
        DefaultStyle = 1 << 0,
        MiddleAnchor = 1 << 1,
        EndAnchor = 1 << 2,
        RightToLeftText = 1 << 3,
        VerticalText = 1 << 4,
        LengthAdjustSpacing = 1 << 5,
        LengthAdjustSpacingAndGlyphs = 1 << 6
    };

    SVGTextChunk(const Vector<InlineIterator::SVGTextBoxIterator>&, unsigned first, unsigned limit, SVGTextFragmentMap&);

    unsigned totalCharacters() const;
    float totalLength() const;
    float totalAnchorShift() const;
    void layout(SVGChunkTransformMap&) const;

private:
    void processTextAnchorCorrection() const;
    void buildBoxTransformations(SVGChunkTransformMap&) const;
    void processTextLengthSpacingCorrection() const;

    bool isVerticalText() const { return m_chunkStyle & VerticalText; }
    float desiredTextLength() const { return m_desiredTextLength; }

    bool hasDesiredTextLength() const { return m_desiredTextLength > 0 && ((m_chunkStyle & LengthAdjustSpacing) || (m_chunkStyle & LengthAdjustSpacingAndGlyphs)); }
    bool hasTextAnchor() const {  return m_chunkStyle & RightToLeftText ? !(m_chunkStyle & EndAnchor) : (m_chunkStyle & (MiddleAnchor | EndAnchor)); }
    bool hasLengthAdjustSpacing() const { return m_chunkStyle & LengthAdjustSpacing; }
    bool hasLengthAdjustSpacingAndGlyphs() const { return m_chunkStyle & LengthAdjustSpacingAndGlyphs; }

    bool boxSpacingAndGlyphsTransform(const Vector<SVGTextFragment>&, AffineTransform&) const;

    Vector<SVGTextFragment>& fragments(InlineIterator::SVGTextBoxIterator);
    const Vector<SVGTextFragment>& fragments(InlineIterator::SVGTextBoxIterator) const;

private:
    // Contains all SVGInlineTextBoxes this chunk spans.
    struct BoxAndFragments {
        InlineIterator::SVGTextBoxIterator box;
        Vector<SVGTextFragment>& fragments;
    };
    Vector<BoxAndFragments> m_boxes;

    unsigned m_chunkStyle { DefaultStyle };
    float m_desiredTextLength { 0 };
};

} // namespace WebCore
