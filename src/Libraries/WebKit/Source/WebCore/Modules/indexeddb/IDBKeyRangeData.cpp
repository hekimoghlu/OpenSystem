/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#include "IDBKeyRangeData.h"

#include "IDBKey.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

IDBKeyRangeData::IDBKeyRangeData(IDBKey* key)
    : lowerKey(key)
    , upperKey(key)
{
}

IDBKeyRangeData::IDBKeyRangeData(const IDBKeyData& keyData)
    : lowerKey(keyData)
    , upperKey(keyData)
{
}

IDBKeyRangeData IDBKeyRangeData::isolatedCopy() const
{
    IDBKeyRangeData result;

    result.lowerKey = lowerKey.isolatedCopy();
    result.upperKey = upperKey.isolatedCopy();
    result.lowerOpen = lowerOpen;
    result.upperOpen = upperOpen;

    return result;
}

bool IDBKeyRangeData::isExactlyOneKey() const
{
    if (isNull() || lowerOpen || upperOpen || !upperKey.isValid() || !lowerKey.isValid())
        return false;

    return !lowerKey.compare(upperKey);
}

bool IDBKeyRangeData::containsKey(const IDBKeyData& key) const
{
    if (lowerKey.isValid()) {
        auto compare = lowerKey.compare(key);
        if (compare > 0)
            return false;
        if (lowerOpen && !compare)
            return false;
    }
    if (upperKey.isValid()) {
        auto compare = upperKey.compare(key);
        if (compare < 0)
            return false;
        if (upperOpen && !compare)
            return false;
    }

    return true;
}

bool IDBKeyRangeData::isValid() const
{
    if (isNull())
        return false;

    if (!lowerKey.isValid() && !lowerKey.isNull())
        return false;

    if (!upperKey.isValid() && !upperKey.isNull())
        return false;

    return true;
}

#if !LOG_DISABLED
String IDBKeyRangeData::loggingString() const
{
    auto result = makeString(lowerOpen ? "( "_s : "[ "_s, lowerKey.loggingString(), ", "_s, upperKey.loggingString(), upperOpen ? " )"_s : " ]"_s);
    if (result.length() > 400)
        result = makeString(StringView(result).left(397), "..."_s);

    return result;
}
#endif

} // namespace WebCore
