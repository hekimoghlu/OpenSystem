/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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

#include "Color.h"
#include "DeprecatedCSSOMPrimitiveValue.h"
#include <wtf/Ref.h>

namespace WebCore {

class DeprecatedCSSOMRGBColor final : public RefCounted<DeprecatedCSSOMRGBColor> {
public:
    static Ref<DeprecatedCSSOMRGBColor> create(CSSStyleDeclaration& owner, const WebCore::Color& color)
    {
        return adoptRef(*new DeprecatedCSSOMRGBColor(owner, color));
    }

    DeprecatedCSSOMPrimitiveValue& red() { return m_red; }
    DeprecatedCSSOMPrimitiveValue& green() { return m_green; }
    DeprecatedCSSOMPrimitiveValue& blue() { return m_blue; }
    DeprecatedCSSOMPrimitiveValue& alpha() { return m_alpha; }

    ResolvedColorType<SRGBA<uint8_t>> color() const { return m_color; }

private:
    template<typename NumberType> static Ref<DeprecatedCSSOMPrimitiveValue> createWrapper(CSSStyleDeclaration& owner, NumberType number)
    {
        return DeprecatedCSSOMPrimitiveValue::create(CSSPrimitiveValue::create(number), owner);
    }

    DeprecatedCSSOMRGBColor(CSSStyleDeclaration& owner, const WebCore::Color& color)
        : m_color(color.toColorTypeLossy<SRGBA<uint8_t>>().resolved())
        , m_red(createWrapper(owner, m_color.red))
        , m_green(createWrapper(owner, m_color.green))
        , m_blue(createWrapper(owner, m_color.blue))
        , m_alpha(createWrapper(owner, color.alphaAsFloat()))
    {
    }

    ResolvedColorType<SRGBA<uint8_t>> m_color;
    Ref<DeprecatedCSSOMPrimitiveValue> m_red;
    Ref<DeprecatedCSSOMPrimitiveValue> m_green;
    Ref<DeprecatedCSSOMPrimitiveValue> m_blue;
    Ref<DeprecatedCSSOMPrimitiveValue> m_alpha;
};

} // namespace WebCore
