/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 30, 2023.
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

#include "SVGTextMetricsBuilder.h"

namespace WebCore {

class RenderBoxModelObject;
class RenderObject;
class RenderSVGInlineText;
class RenderSVGText;
class SVGTextPositioningElement;

// SVGTextLayoutAttributesBuilder performs the first layout phase for SVG text.
//
// It extracts the x/y/dx/dy/rotate values from the SVGTextPositioningElements in the DOM.
// These values are propagated to the corresponding RenderSVGInlineText renderers.
// The first layout phase only extracts the relevant information needed in RenderBlockLineLayout
// to create the InlineBox tree based on text chunk boundaries & BiDi information.
// The second layout phase is carried out by SVGTextLayoutEngine.

class SVGTextLayoutAttributesBuilder {
    WTF_MAKE_NONCOPYABLE(SVGTextLayoutAttributesBuilder);
public:
    SVGTextLayoutAttributesBuilder();
    bool buildLayoutAttributesForForSubtree(RenderSVGText&);
    void buildLayoutAttributesForTextRenderer(RenderSVGInlineText&);

    void rebuildMetricsForSubtree(RenderSVGText&);

    // Invoked whenever the underlying DOM tree changes, so that m_textPositions is rebuild.
    void clearTextPositioningElements() { m_textPositions.clear(); }
    unsigned numberOfTextPositioningElements() const { return m_textPositions.size(); }

private:
    struct TextPosition {
        TextPosition(SVGTextPositioningElement* newElement = 0, unsigned newStart = 0, unsigned newLength = 0)
            : element(newElement)
            , start(newStart)
            , length(newLength)
        {
        }

        SVGTextPositioningElement* element;
        unsigned start;
        unsigned length;
    };

    void buildCharacterDataMap(RenderSVGText&);
    void collectTextPositioningElements(RenderBoxModelObject&, bool& lastCharacterWasSpace);
    void fillCharacterDataMap(const TextPosition&);

private:
    unsigned m_textLength;
    Vector<TextPosition> m_textPositions;
    SVGCharacterDataMap m_characterDataMap;
    SVGTextMetricsBuilder m_metricsBuilder;
};

} // namespace WebCore
