/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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

#include "SVGTextChunk.h"
#include <wtf/Vector.h>

namespace WebCore {

class RenderSVGInlineText;
class SVGInlineTextBox;
struct SVGTextFragment;

// SVGTextChunkBuilder performs the third layout phase for SVG text.
//
// Phase one built the layout information from the SVG DOM stored in the RenderSVGInlineText objects (SVGTextLayoutAttributes).
// Phase two performed the actual per-character layout, computing the final positions for each character, stored in the SVGInlineTextBox objects (SVGTextFragment).
// Phase three performs all modifications that have to be applied to each individual text chunk (text-anchor & textLength).

class SVGTextChunkBuilder {
public:
    SVGTextChunkBuilder();
    SVGTextChunkBuilder(SVGTextChunkBuilder&&) = default;
    SVGTextChunkBuilder(const SVGTextChunkBuilder&) = delete;

    const Vector<SVGTextChunk>& textChunks() const { return m_textChunks; }
    unsigned totalCharacters() const;
    float totalLength() const;
    float totalAnchorShift() const;
    AffineTransform transformationForTextBox(InlineIterator::SVGTextBoxIterator) const;

    void buildTextChunks(const Vector<InlineIterator::SVGTextBoxIterator>& lineLayoutBoxes, const UncheckedKeyHashSet<InlineIterator::SVGTextBox::Key>& chunkStarts, SVGTextFragmentMap&);
    void layoutTextChunks(const Vector<InlineIterator::SVGTextBoxIterator>& lineLayoutBoxes, const UncheckedKeyHashSet<InlineIterator::SVGTextBox::Key>& chunkStarts, SVGTextFragmentMap&);

private:
    Vector<SVGTextChunk> m_textChunks;
    SVGChunkTransformMap m_textBoxTransformations;
};

} // namespace WebCore
