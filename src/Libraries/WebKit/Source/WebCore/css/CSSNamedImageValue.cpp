/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#include "CSSNamedImageValue.h"

#include "StyleNamedImage.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

CSSNamedImageValue::CSSNamedImageValue(String&& name)
    : CSSValue { ClassType::NamedImage }
    , m_name { WTFMove(name) }
{
}

CSSNamedImageValue::~CSSNamedImageValue() = default;

String CSSNamedImageValue::customCSSText() const
{
    return makeString("-webkit-named-image("_s, m_name, ')');
}

bool CSSNamedImageValue::equals(const CSSNamedImageValue& other) const
{
    return m_name == other.m_name;
}

RefPtr<StyleImage> CSSNamedImageValue::createStyleImage(const Style::BuilderState&) const
{
    if (m_cachedStyleImage)
        return m_cachedStyleImage;

    m_cachedStyleImage = StyleNamedImage::create(m_name);
    return m_cachedStyleImage;
}

} // namespace WebCore
