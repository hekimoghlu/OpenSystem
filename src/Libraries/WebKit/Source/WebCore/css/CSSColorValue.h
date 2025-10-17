/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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

#include "CSSColor.h"
#include "CSSValue.h"

namespace WebCore {

class Color;

class CSSColorValue final : public CSSValue {
public:
    static Ref<CSSColorValue> create(CSS::Color);
    static Ref<CSSColorValue> create(WebCore::Color);

    const CSS::Color& color() const { return m_color; }

    String customCSSText() const;
    bool equals(const CSSColorValue&) const;
    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>&) const;

    WEBCORE_EXPORT static WebCore::Color absoluteColor(const CSSValue&);

private:
    friend class CSSValuePool;
    friend class StaticCSSValuePool;
    friend LazyNeverDestroyed<CSSColorValue>;

    CSSColorValue(CSS::Color);
    CSSColorValue(WebCore::Color);
    CSSColorValue(StaticCSSValueTag, WebCore::Color);

    CSS::Color m_color;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSColorValue, isColor())
