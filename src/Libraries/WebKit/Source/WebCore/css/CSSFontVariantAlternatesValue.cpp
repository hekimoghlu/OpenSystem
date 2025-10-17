/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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
#include "CSSFontVariantAlternatesValue.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

CSSFontVariantAlternatesValue::CSSFontVariantAlternatesValue(FontVariantAlternates&& alternates)
    : CSSValue(ClassType::FontVariantAlternates)
    , m_value(alternates)
{
}

String CSSFontVariantAlternatesValue::customCSSText() const
{
    TextStream ts;
    // For the moment, the stream operator implements the CSS serialization exactly.
    // If it changes for whatever reason, we should reimplement the CSS serialization here.
    ts << m_value;
    return ts.release();
}

bool CSSFontVariantAlternatesValue::equals(const CSSFontVariantAlternatesValue& other) const
{
    return value() == other.value();
}

}
