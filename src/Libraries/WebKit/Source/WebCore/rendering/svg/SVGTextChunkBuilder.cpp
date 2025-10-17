/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 21, 2021.
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
#include "config.h"
#include "SVGTextChunkBuilder.h"

#include "SVGElement.h"
#include "SVGInlineTextBox.h"
#include "SVGLengthContext.h"
#include "SVGTextFragment.h"

namespace WebCore {

SVGTextChunkBuilder::SVGTextChunkBuilder()
{
}

unsigned SVGTextChunkBuilder::totalCharacters() const
{
    unsigned characters = 0;
    for (const auto& chunk : m_textChunks)
        characters += chunk.totalCharacters();
    return characters;
}

float SVGTextChunkBuilder::totalLength() const
{
    float length = 0;
    for (const auto& chunk : m_textChunks)
        length += chunk.totalLength();
    return length;
}

float SVGTextChunkBuilder::totalAnchorShift() const
{
    float anchorShift = 0;
    for (const auto& chunk : m_textChunks)
        anchorShift += chunk.totalAnchorShift();
    return anchorShift;
}

AffineTransform SVGTextChunkBuilder::transformationForTextBox(InlineIterator::SVGTextBoxIterator textBox) const
{
    auto it = m_textBoxTransformations.find(makeKey(*textBox));
    return it == m_textBoxTransformations.end() ? AffineTransform() : it->value;
}

void SVGTextChunkBuilder::buildTextChunks(const Vector<InlineIterator::SVGTextBoxIterator>& lineLayoutBoxes, const UncheckedKeyHashSet<InlineIterator::SVGTextBox::Key>& chunkStarts, SVGTextFragmentMap& fragmentMap)
{
    if (lineLayoutBoxes.isEmpty())
        return;

    unsigned limit = lineLayoutBoxes.size();
    unsigned first = limit;

    for (unsigned i = 0; i < limit; ++i) {
        if (!chunkStarts.contains(makeKey(*lineLayoutBoxes[i])))
            continue;

        if (first == limit)
            first = i;
        else {
            ASSERT_WITH_SECURITY_IMPLICATION(first != i);
            m_textChunks.append(SVGTextChunk(lineLayoutBoxes, first, i, fragmentMap));
            first = i;
        }
    }

    if (first != limit)
        m_textChunks.append(SVGTextChunk(lineLayoutBoxes, first, limit, fragmentMap));
}

void SVGTextChunkBuilder::layoutTextChunks(const Vector<InlineIterator::SVGTextBoxIterator>& lineLayoutBoxes, const UncheckedKeyHashSet<InlineIterator::SVGTextBox::Key>& chunkStarts, SVGTextFragmentMap& fragmentMap)
{
    buildTextChunks(lineLayoutBoxes, chunkStarts, fragmentMap);
    if (m_textChunks.isEmpty())
        return;

    for (const auto& chunk : m_textChunks)
        chunk.layout(m_textBoxTransformations);

    m_textChunks.clear();
}

}
