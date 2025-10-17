/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 19, 2024.
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
#include "CSSCanvasValue.h"

#include "StyleCanvasImage.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

CSSCanvasValue::CSSCanvasValue(String&& name)
    : CSSValue { ClassType::Canvas }
    , m_name { WTFMove(name) }
{
}

CSSCanvasValue::~CSSCanvasValue() = default;

String CSSCanvasValue::customCSSText() const
{
    return makeString("-webkit-canvas("_s, m_name, ')');
}

bool CSSCanvasValue::equals(const CSSCanvasValue& other) const
{
    return m_name == other.m_name;
}

RefPtr<StyleImage> CSSCanvasValue::createStyleImage(const Style::BuilderState&) const
{
    if (m_cachedStyleImage)
        return m_cachedStyleImage;

    m_cachedStyleImage = StyleCanvasImage::create(m_name);
    return m_cachedStyleImage;
}

} // namespace WebCore
