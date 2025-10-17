/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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
#include "CSSLineBoxContainValue.h"

#include <wtf/text/StringBuilder.h>

namespace WebCore {

CSSLineBoxContainValue::CSSLineBoxContainValue(OptionSet<LineBoxContain> value)
    : CSSValue(ClassType::LineBoxContain)
    , m_value(value)
{
}

String CSSLineBoxContainValue::customCSSText() const
{
    StringBuilder text;
    if (m_value.contains(LineBoxContain::Block))
        text.append("block"_s);
    if (m_value.contains(LineBoxContain::Inline))
        text.append(text.isEmpty() ? ""_s : " "_s, "inline"_s);
    if (m_value.contains(LineBoxContain::Font))
        text.append(text.isEmpty() ? ""_s : " "_s, "font"_s);
    if (m_value.contains(LineBoxContain::Glyphs))
        text.append(text.isEmpty() ? ""_s : " "_s, "glyphs"_s);
    if (m_value.contains(LineBoxContain::Replaced))
        text.append(text.isEmpty() ? ""_s : " "_s, "replaced"_s);
    if (m_value.contains(LineBoxContain::InlineBox))
        text.append(text.isEmpty() ? ""_s : " "_s, "inline-box"_s);
    if (m_value.contains(LineBoxContain::InitialLetter))
        text.append(text.isEmpty() ? ""_s : " "_s, "initial-letter"_s);
    return text.toString();
}

}
