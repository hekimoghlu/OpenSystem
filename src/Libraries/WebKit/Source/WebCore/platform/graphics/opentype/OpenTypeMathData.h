/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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

#include "Glyph.h"
#include <wtf/Forward.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

#if USE(HARFBUZZ)
#include "HbUniquePtr.h"
#include <hb-ot.h>
#endif

namespace WebCore {

class FontPlatformData;
class SharedBuffer;
class Font;

class OpenTypeMathData : public RefCounted<OpenTypeMathData> {
public:
    static Ref<OpenTypeMathData> create(const FontPlatformData& font)
    {
        return adoptRef(*new OpenTypeMathData(font));
    }
    ~OpenTypeMathData();

#if ENABLE(OPENTYPE_MATH)
    bool hasMathData() const { return m_mathBuffer; }
#elif USE(HARFBUZZ)
    bool hasMathData() const { return m_mathFont.get(); }
#else
    bool hasMathData() const { return false; }
#endif

    // These constants are defined in the MATH table.
    // The implementation of OpenTypeMathData::getMathConstant assumes that they correspond to the indices of the MathContant table.
    enum MathConstant {
        ScriptPercentScaleDown,
        ScriptScriptPercentScaleDown,
        DelimitedSubFormulaMinHeight,
        DisplayOperatorMinHeight,
        MathLeading,
        AxisHeight,
        AccentBaseHeight,
        FlattenedAccentBaseHeight,
        SubscriptShiftDown,
        SubscriptTopMax,
        SubscriptBaselineDropMin,
        SuperscriptShiftUp,
        SuperscriptShiftUpCramped,
        SuperscriptBottomMin,
        SuperscriptBaselineDropMax,
        SubSuperscriptGapMin,
        SuperscriptBottomMaxWithSubscript,
        SpaceAfterScript,
        UpperLimitGapMin,
        UpperLimitBaselineRiseMin,
        LowerLimitGapMin,
        LowerLimitBaselineDropMin,
        StackTopShiftUp,
        StackTopDisplayStyleShiftUp,
        StackBottomShiftDown,
        StackBottomDisplayStyleShiftDown,
        StackGapMin,
        StackDisplayStyleGapMin,
        StretchStackTopShiftUp,
        StretchStackBottomShiftDown,
        StretchStackGapAboveMin,
        StretchStackGapBelowMin,
        FractionNumeratorShiftUp,
        FractionNumeratorDisplayStyleShiftUp,
        FractionDenominatorShiftDown,
        FractionDenominatorDisplayStyleShiftDown,
        FractionNumeratorGapMin,
        FractionNumDisplayStyleGapMin,
        FractionRuleThickness,
        FractionDenominatorGapMin,
        FractionDenomDisplayStyleGapMin,
        SkewedFractionHorizontalGap,
        SkewedFractionVerticalGap,
        OverbarVerticalGap,
        OverbarRuleThickness,
        OverbarExtraAscender,
        UnderbarVerticalGap,
        UnderbarRuleThickness,
        UnderbarExtraDescender,
        RadicalVerticalGap,
        RadicalDisplayStyleVerticalGap,
        RadicalRuleThickness,
        RadicalExtraAscender,
        RadicalKernBeforeDegree,
        RadicalKernAfterDegree,
        RadicalDegreeBottomRaisePercent
    };

    struct AssemblyPart {
        Glyph glyph;
        bool isExtender;
    };

    float getMathConstant(const Font&, MathConstant) const;
    float getItalicCorrection(const Font&, Glyph) const;
    void getMathVariants(Glyph, bool isVertical, Vector<Glyph>& sizeVariants, Vector<AssemblyPart>& assemblyParts) const;

private:
    explicit OpenTypeMathData(const FontPlatformData&);

#if ENABLE(OPENTYPE_MATH)
    RefPtr<SharedBuffer> m_mathBuffer;
#elif USE(HARFBUZZ)
    HbUniquePtr<hb_font_t> m_mathFont;
#endif
};

} // namespace WebCore

#endif // ENABLE(MATHML)
