/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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

#include "CSSFilterProperty.h"
#include "CSSValue.h"
#include <wtf/Function.h>

namespace WebCore {

namespace Style {
class BuilderState;
}

class StyleImage;

class CSSFilterImageValue final : public CSSValue {
public:
    static Ref<CSSFilterImageValue> create(Ref<CSSValue>&& imageValueOrNone, CSS::FilterProperty&& filter)
    {
        return adoptRef(*new CSSFilterImageValue(WTFMove(imageValueOrNone), WTFMove(filter)));
    }
    ~CSSFilterImageValue();

    bool equals(const CSSFilterImageValue&) const;
    bool equalInputImages(const CSSFilterImageValue&) const;

    String customCSSText() const;
    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>&) const;

    RefPtr<StyleImage> createStyleImage(const Style::BuilderState&) const;

private:
    explicit CSSFilterImageValue(Ref<CSSValue>&& imageValueOrNone, CSS::FilterProperty&&);

    Ref<CSSValue> m_imageValueOrNone;
    CSS::FilterProperty m_filter;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSFilterImageValue, isFilterImageValue())
