/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#include "Path.h"
#include "SVGTextChunkBuilder.h"
#include "SVGTextFragment.h"
#include "SVGTextLayoutAttributes.h"

namespace WebCore {

class RenderObject;
class RenderStyle;
class RenderSVGInlineText;
class RenderSVGTextPath;
class SVGElement;
class SVGInlineTextBox;
class SVGRenderStyle;

// SVGTextLayoutEngine performs the second layout phase for SVG text.
//
// The InlineBox tree was created, containing the text chunk information, necessary to apply
// certain SVG specific text layout properties (text-length adjustments and text-anchor).
// The second layout phase uses the SVGTextLayoutAttributes stored in the individual
// RenderSVGInlineText renderers to compute the final positions for each character
// which are stored in the SVGInlineTextBox objects.

class SVGTextLayoutEngine {
public:
    SVGTextLayoutEngine(Vector<SVGTextLayoutAttributes*>&);
    SVGTextLayoutEngine(SVGTextLayoutEngine&&) = default;
    SVGTextLayoutEngine(const SVGTextLayoutEngine&) = delete;

    Vector<SVGTextLayoutAttributes*>& layoutAttributes() { return m_layoutAttributes; }

    void beginTextPathLayout(const RenderSVGTextPath&, SVGTextLayoutEngine& lineLayout);
    void endTextPathLayout();

    void layoutInlineTextBox(InlineIterator::SVGTextBoxIterator);

    SVGTextFragmentMap finishLayout();

private:
    void updateCharacterPositionIfNeeded(float& x, float& y);
    void updateCurrentTextPosition(float x, float y, float glyphAdvance);
    void updateRelativePositionAdjustmentsIfNeeded(float dx, float dy);

    void recordTextFragment(InlineIterator::SVGTextBoxIterator, const Vector<SVGTextMetrics>&);
    bool parentDefinesTextLength(RenderObject*) const;

    void layoutTextOnLineOrPath(InlineIterator::SVGTextBoxIterator, const RenderSVGInlineText&, const RenderStyle&);
    void finalizeTransformMatrices(Vector<InlineIterator::SVGTextBoxIterator>&);

    bool currentLogicalCharacterAttributes(SVGTextLayoutAttributes*&);
    bool currentLogicalCharacterMetrics(SVGTextLayoutAttributes*&, SVGTextMetrics&);
    bool currentVisualCharacterMetrics(const InlineIterator::SVGTextBox&, const Vector<SVGTextMetrics>&, SVGTextMetrics&);

    void advanceToNextLogicalCharacter(const SVGTextMetrics&);
    void advanceToNextVisualCharacter(const SVGTextMetrics&);

private:
    Vector<SVGTextLayoutAttributes*>& m_layoutAttributes;

    Vector<InlineIterator::SVGTextBoxIterator> m_lineLayoutBoxes;
    Vector<InlineIterator::SVGTextBoxIterator> m_pathLayoutBoxes;

    // Output.
    UncheckedKeyHashMap<InlineIterator::SVGTextBox::Key, Vector<SVGTextFragment>> m_fragmentMap;

    SVGTextChunkBuilder m_chunkLayoutBuilder;
    UncheckedKeyHashSet<InlineIterator::SVGTextBox::Key> m_lineLayoutChunkStarts;

    SVGTextFragment m_currentTextFragment;
    unsigned m_layoutAttributesPosition { 0 };
    unsigned m_logicalCharacterOffset { 0 };
    unsigned m_logicalMetricsListOffset { 0 };
    unsigned m_visualCharacterOffset { 0 };
    unsigned m_visualMetricsListOffset { 0 };
    float m_x { 0.0f };
    float m_y { 0.0f };
    float m_dx { 0.0f };
    float m_dy { 0.0f };
    float m_lastChunkStartPosition { 0.0f };
    bool m_lastChunkHasTextLength { false };
    bool m_lastChunkIsVerticalText { false };
    bool m_isVerticalText { false };
    bool m_inPathLayout { false };

    // Text on path layout
    Path m_textPath;
    float m_textPathLength { 0.0f };
    float m_textPathStartOffset { 0.0f };
    float m_textPathCurrentOffset { 0.0f };
    float m_textPathSpacing { 0.0f };
    float m_textPathScaling { 1.0f };
};

} // namespace WebCore
