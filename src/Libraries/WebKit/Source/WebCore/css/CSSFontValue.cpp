/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
#include "CSSFontValue.h"

#include "CSSFontStyleWithAngleValue.h"
#include "CSSValueList.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

String CSSFontValue::customCSSText() const
{
    // font variant weight size / line-height family
    StringBuilder result;
    if (style)
        result.append(style->cssText());
    if (variant)
        result.append(result.isEmpty() ? ""_s : " "_s, variant->cssText());
    if (weight)
        result.append(result.isEmpty() ? ""_s : " "_s, weight->cssText());
    if (width)
        result.append(result.isEmpty() ? ""_s : " "_s, width->cssText());
    if (size)
        result.append(result.isEmpty() ? ""_s : " "_s, size->cssText());
    if (lineHeight)
        result.append(size ? " / "_s : result.isEmpty() ? ""_s : " "_s, lineHeight->cssText());
    if (family)
        result.append(result.isEmpty() ? ""_s : " "_s, family->cssText());
    return result.toString();
}

bool CSSFontValue::equals(const CSSFontValue& other) const
{
    return compareCSSValuePtr(style, other.style)
        && compareCSSValuePtr(variant, other.variant)
        && compareCSSValuePtr(weight, other.weight)
        && compareCSSValuePtr(width, other.width)
        && compareCSSValuePtr(size, other.size)
        && compareCSSValuePtr(lineHeight, other.lineHeight)
        && compareCSSValuePtr(family, other.family);
}

IterationStatus CSSFontValue::customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
{
    if (style) {
        if (func(*style) == IterationStatus::Done)
            return IterationStatus::Done;
    }
    if (variant) {
        if (func(*variant) == IterationStatus::Done)
            return IterationStatus::Done;
    }
    if (weight) {
        if (func(*weight) == IterationStatus::Done)
            return IterationStatus::Done;
    }
    if (width) {
        if (func(*width) == IterationStatus::Done)
            return IterationStatus::Done;
    }
    if (size) {
        if (func(*size) == IterationStatus::Done)
            return IterationStatus::Done;
    }
    if (lineHeight) {
        if (func(*lineHeight) == IterationStatus::Done)
            return IterationStatus::Done;
    }
    if (family) {
        if (func(*family) == IterationStatus::Done)
            return IterationStatus::Done;
    }
    return IterationStatus::Continue;
}

}
