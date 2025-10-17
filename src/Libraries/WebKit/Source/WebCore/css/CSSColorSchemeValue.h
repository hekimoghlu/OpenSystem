/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 18, 2022.
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

#include "CSSColorScheme.h"
#include "CSSValue.h"

#if ENABLE(DARK_MODE_CSS)

namespace WebCore {

class CSSColorSchemeValue final : public CSSValue {
public:
    static Ref<CSSColorSchemeValue> create(CSS::ColorScheme);

    String customCSSText() const;
    bool equals(const CSSColorSchemeValue&) const;
    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>&) const;

    const CSS::ColorScheme& colorScheme() const { return m_colorScheme; }

private:
    CSSColorSchemeValue(CSS::ColorScheme);

    CSS::ColorScheme m_colorScheme;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSColorSchemeValue, isColorScheme())

#endif
