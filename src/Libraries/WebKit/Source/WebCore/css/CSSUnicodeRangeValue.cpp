/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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
#include "CSSUnicodeRangeValue.h"

#include <wtf/HexNumber.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

String CSSUnicodeRangeValue::customCSSText() const
{
    if (m_from == m_to)
        return makeString("U+"_s, hex(m_from, Lowercase));

    return makeString("U+"_s, hex(m_from, Lowercase), '-', hex(m_to, Lowercase));
}

bool CSSUnicodeRangeValue::equals(const CSSUnicodeRangeValue& other) const
{
    return m_from == other.m_from && m_to == other.m_to;
}

}
