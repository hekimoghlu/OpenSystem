/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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

#include "SVGTextLayoutAttributes.h"
#include "TextRun.h"
#include "WidthIterator.h"

namespace WebCore {

class RenderElement;
class RenderSVGInlineText;
class RenderSVGText;
struct MeasureTextData;

class SVGTextMetricsBuilder {
    WTF_MAKE_NONCOPYABLE(SVGTextMetricsBuilder);
public:
    SVGTextMetricsBuilder();
    void measureTextRenderer(RenderSVGText&, RenderSVGInlineText* stopAtLeaf);
    void buildMetricsAndLayoutAttributes(RenderSVGText&, RenderSVGInlineText* stopAtLeaf, SVGCharacterDataMap& allCharactersMap);

private:
    bool advance();
    void advanceSimpleText();
    void advanceComplexText();
    bool currentCharacterStartsSurrogatePair() const;

    void initializeMeasurementWithTextRenderer(RenderSVGInlineText&);
    void walkTree(RenderElement&, RenderSVGInlineText* stopAtLeaf, MeasureTextData&);
    std::tuple<unsigned, UChar> measureTextRenderer(RenderSVGInlineText&, const MeasureTextData&, std::tuple<unsigned, UChar>);

    SingleThreadWeakPtr<RenderSVGInlineText> m_text;
    TextRun m_run;
    unsigned m_textPosition;
    bool m_isComplexText { false };
    bool m_canUseSimplifiedTextMeasuring { false };
    SVGTextMetrics m_currentMetrics;
    float m_totalWidth;

    // Simple text only.
    std::unique_ptr<WidthIterator> m_simpleWidthIterator;

    // Complex text only.
    SVGTextMetrics m_complexStartToCurrentMetrics;
};

} // namespace WebCore
