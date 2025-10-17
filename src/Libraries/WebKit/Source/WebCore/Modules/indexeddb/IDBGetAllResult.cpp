/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#include "IDBGetAllResult.h"

#include <wtf/CrossThreadCopier.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBGetAllResult);

IDBGetAllResult::IDBGetAllResult(const IDBGetAllResult& that, IsolatedCopyTag)
{
    isolatedCopy(that, *this);
}

IDBGetAllResult IDBGetAllResult::isolatedCopy() const
{
    return { *this, IsolatedCopy };
}

void IDBGetAllResult::isolatedCopy(const IDBGetAllResult& source, IDBGetAllResult& destination)
{
    destination.m_type = source.m_type;
    destination.m_keys = crossThreadCopy(source.m_keys);
    destination.m_values = crossThreadCopy(source.m_values);
    destination.m_keyPath = crossThreadCopy(source.m_keyPath);
}

void IDBGetAllResult::addKey(IDBKeyData&& key)
{
    m_keys.append(WTFMove(key));
}

void IDBGetAllResult::addValue(IDBValue&& value)
{
    m_values.append(WTFMove(value));
}

const Vector<IDBKeyData>& IDBGetAllResult::keys() const
{
    return m_keys;
}

const Vector<IDBValue>& IDBGetAllResult::values() const
{
    return m_values;
}

Vector<String> IDBGetAllResult::allBlobFilePaths() const
{
    ASSERT(m_type == IndexedDB::GetAllType::Values);

    HashSet<String> pathSet;
    for (auto& value : m_values) {
        for (auto& path : value.blobFilePaths())
            pathSet.add(path);
    }

    return copyToVector(pathSet);
}

} // namespace WebCore
