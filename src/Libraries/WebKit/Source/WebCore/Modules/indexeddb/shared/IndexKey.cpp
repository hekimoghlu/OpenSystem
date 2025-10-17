/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#include "IndexKey.h"

#include <wtf/CrossThreadCopier.h>

namespace WebCore {

IndexKey::IndexKey() = default;

IndexKey::IndexKey(Data&& keys)
    : m_keys(WTFMove(keys))
{
}

IndexKey IndexKey::isolatedCopy() const &
{
    return { crossThreadCopy(m_keys) };
}

IndexKey IndexKey::isolatedCopy() &&
{
    return { crossThreadCopy(WTFMove(m_keys)) };
}

IDBKeyData IndexKey::asOneKey() const
{
    return WTF::switchOn(m_keys, [](std::nullptr_t) {
        return IDBKeyData { };
    }, [](const IDBKeyData& keyData) {
        return keyData;
    }, [](const Vector<IDBKeyData>& keyDataVector) {
        IDBKeyData result;
        result.setArrayValue(keyDataVector);
        return result;
    });
}

Vector<IDBKeyData> IndexKey::multiEntry() const
{
    Vector<IDBKeyData> multiEntry;

    WTF::switchOn(m_keys, [](std::nullptr_t) {
    }, [&](const IDBKeyData& keyData) {
        if (keyData.isValid())
            multiEntry.append(keyData);
    }, [&](const Vector<IDBKeyData>& keyDataVector) {
        for (auto& key : keyDataVector) {
            if (!key.isValid())
                continue;

            bool skip = false;
            for (auto& otherKey : multiEntry) {
                if (key == otherKey) {
                    skip = true;
                    break;
                }
            }

            if (!skip)
                multiEntry.append(key);
        }
    });

    return multiEntry;
}

} // namespace WebCore
