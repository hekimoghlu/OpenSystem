/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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

#include "IDBKeyData.h"
#include "IDBKeyRange.h"

namespace WebCore {

class IDBKey;

struct IDBKeyRangeData {
    IDBKeyRangeData()
    {
    }

    static IDBKeyRangeData allKeys()
    {
        IDBKeyRangeData result;
        result.lowerKey = IDBKeyData::minimum();
        result.upperKey = IDBKeyData::maximum();
        return result;
    }

    IDBKeyRangeData(IDBKey*);
    IDBKeyRangeData(const IDBKeyData&);

    IDBKeyRangeData(IDBKeyRange* keyRange)
    {
        if (!keyRange)
            return;

        lowerKey = keyRange->lower();
        upperKey = keyRange->upper();
        lowerOpen = keyRange->lowerOpen();
        upperOpen = keyRange->upperOpen();
    }

    IDBKeyRangeData(IDBKeyData&& lowerKey, IDBKeyData&& upperKey, bool lowerOpen, bool upperOpen)
        : lowerKey(WTFMove(lowerKey))
        , upperKey(WTFMove(upperKey))
        , lowerOpen(WTFMove(lowerOpen))
        , upperOpen(WTFMove(upperOpen))
    {
    }

    WEBCORE_EXPORT IDBKeyRangeData isolatedCopy() const;

    WEBCORE_EXPORT bool isExactlyOneKey() const;
    bool containsKey(const IDBKeyData&) const;
    bool isValid() const;

    IDBKeyData lowerKey;
    IDBKeyData upperKey;

    bool lowerOpen { false };
    bool upperOpen { false };

    bool isNull() const { return lowerKey.isNull() && upperKey.isNull(); };

#if !LOG_DISABLED
    String loggingString() const;
#endif
};

} // namespace WebCore
