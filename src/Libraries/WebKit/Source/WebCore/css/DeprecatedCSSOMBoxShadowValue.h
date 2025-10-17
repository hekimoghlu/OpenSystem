/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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

#include "CSSBoxShadow.h"
#include "DeprecatedCSSOMValue.h"

namespace WebCore {

// This class is needed to maintain compatibility with the historical CSSOM representation of the `box-shadow` related properties.
// It should be used only as an element in a DeprecatedCSSOMValueList created by CSSBoxShadowPropertyValue.
class DeprecatedCSSOMBoxShadowValue final : public DeprecatedCSSOMValue {
public:
    static Ref<DeprecatedCSSOMBoxShadowValue> create(CSS::BoxShadow, CSSStyleDeclaration&);

    String cssText() const;
    unsigned short cssValueType() const { return CSS_CUSTOM; }

private:
    DeprecatedCSSOMBoxShadowValue(CSS::BoxShadow&&, CSSStyleDeclaration&);

    CSS::BoxShadow m_shadow;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSSOM_VALUE(DeprecatedCSSOMBoxShadowValue, isBoxShadowValue())
