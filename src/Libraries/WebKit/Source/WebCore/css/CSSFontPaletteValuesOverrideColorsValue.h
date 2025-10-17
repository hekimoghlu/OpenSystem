/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 12, 2023.
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

#include "CSSPrimitiveValue.h"
#include "CSSValue.h"
#include "CachedResourceHandle.h"
#include "ResourceLoaderOptions.h"
#include <wtf/Function.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CachedFont;
class FontLoadRequest;
class SVGFontFaceElement;
class ScriptExecutionContext;

class CSSFontPaletteValuesOverrideColorsValue final : public CSSValue {
public:
    static Ref<CSSFontPaletteValuesOverrideColorsValue> create(Ref<CSSPrimitiveValue>&& key, Ref<CSSValue>&& color)
    {
        return adoptRef(*new CSSFontPaletteValuesOverrideColorsValue(WTFMove(key), WTFMove(color)));
    }

    const CSSPrimitiveValue& key() const { return m_key; }
    const CSSValue& color() const { return m_color; }

    String customCSSText() const;

    bool equals(const CSSFontPaletteValuesOverrideColorsValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>& func) const
    {
        if (func(m_key.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        if (func(m_color.get()) == IterationStatus::Done)
            return IterationStatus::Done;
        return IterationStatus::Continue;
    }

private:
    CSSFontPaletteValuesOverrideColorsValue(Ref<CSSPrimitiveValue>&& key, Ref<CSSValue>&& color)
        : CSSValue(ClassType::FontPaletteValuesOverrideColors)
        , m_key(WTFMove(key))
        , m_color(WTFMove(color))
    {
    }

    Ref<CSSPrimitiveValue> m_key;
    Ref<CSSValue> m_color;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSFontPaletteValuesOverrideColorsValue, isFontPaletteValuesOverrideColorsValue())
