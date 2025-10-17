/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 13, 2024.
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
#include "InternalsMapLike.h"

#include "IDLTypes.h"
#include "JSDOMMapLike.h"
#include <wtf/Vector.h>

namespace WebCore {

InternalsMapLike::InternalsMapLike()
{
    m_values.add("init"_s, 0);
}

void InternalsMapLike::initializeMapLike(DOMMapAdapter& map)
{
    for (auto& keyValue : m_values)
        map.set<IDLDOMString, IDLUnsignedLong>(keyValue.key, keyValue.value);
}

void InternalsMapLike::setFromMapLike(String&& key, unsigned value)
{
    m_values.set(WTFMove(key), value);
}

void InternalsMapLike::clear()
{
    m_values.clear();
}

bool InternalsMapLike::remove(const String& key)
{
    return m_values.remove(key);
}

Vector<String> InternalsMapLike::inspectKeys() const
{
    auto result = copyToVector(m_values.keys());
    std::sort(result.begin(), result.end(), WTF::codePointCompareLessThan);
    return result;
}

Vector<unsigned> InternalsMapLike::inspectValues() const
{
    auto result = copyToVector(m_values.values());
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace WebCore
