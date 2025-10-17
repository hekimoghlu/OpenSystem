/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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

#if ENABLE(MATHML)

#include "MathMLScriptsElement.h"
#include "RenderMathMLRow.h"

namespace WebCore {

// Render a base with scripts.
class RenderMathMLScripts : public RenderMathMLRow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderMathMLScripts);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderMathMLScripts);
public:
    RenderMathMLScripts(Type, MathMLScriptsElement&, RenderStyle&&);
    virtual ~RenderMathMLScripts();

    RenderMathMLOperator* unembellishedOperator() const final;

protected:
    bool isRenderMathMLScripts() const override { return true; }
    ASCIILiteral renderName() const override { return "RenderMathMLScripts"_s; }
    MathMLScriptsElement::ScriptType scriptType() const;
    void computePreferredLogicalWidths() override;
    void layoutBlock(bool relayoutChildren, LayoutUnit pageLogicalHeight = 0_lu) override;

private:
    MathMLScriptsElement& element() const;
    std::optional<LayoutUnit> firstLineBaseline() const final;
    struct ReferenceChildren {
        RenderBox* base;
        RenderBox* prescriptDelimiter;
        RenderBox* firstPostScript;
        RenderBox* firstPreScript;
    };
    std::optional<ReferenceChildren> validateAndGetReferenceChildren() const;
    LayoutUnit spaceAfterScript();
    LayoutUnit italicCorrection(const ReferenceChildren&);
    struct VerticalParameters {
        LayoutUnit subscriptShiftDown;
        LayoutUnit superscriptShiftUp;
        LayoutUnit subscriptBaselineDropMin;
        LayoutUnit superScriptBaselineDropMax;
        LayoutUnit subSuperscriptGapMin;
        LayoutUnit superscriptBottomMin;
        LayoutUnit subscriptTopMax;
        LayoutUnit superscriptBottomMaxWithSubscript;
    };
    VerticalParameters verticalParameters() const;
    struct VerticalMetrics {
        LayoutUnit subShift;
        LayoutUnit supShift;
        LayoutUnit ascent;
        LayoutUnit descent;
    };
    VerticalMetrics verticalMetrics(const ReferenceChildren&);
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderMathMLScripts, isRenderMathMLScripts())

#endif // ENABLE(MATHML)
