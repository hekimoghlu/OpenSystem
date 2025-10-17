/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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
#include "IDBValue.h"

#include "SerializedScriptValue.h"
#include <wtf/CrossThreadTask.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IDBValue);

IDBValue::IDBValue()
{
}

IDBValue::IDBValue(const SerializedScriptValue& scriptValue)
    : m_data(ThreadSafeDataBuffer::copyData(scriptValue.wireBytes()))
    , m_blobURLs(scriptValue.blobURLs())
{
}

IDBValue::IDBValue(const ThreadSafeDataBuffer& value)
    : m_data(value)
{
}

IDBValue::IDBValue(const SerializedScriptValue& scriptValue, const Vector<String>& blobURLs, const Vector<String>& blobFilePaths)
    : m_data(ThreadSafeDataBuffer::copyData(scriptValue.wireBytes()))
    , m_blobURLs(blobURLs)
    , m_blobFilePaths(blobFilePaths)
{
    ASSERT(m_data.data());
}

IDBValue::IDBValue(const ThreadSafeDataBuffer& value, Vector<String>&& blobURLs, Vector<String>&& blobFilePaths)
    : m_data(value)
    , m_blobURLs(WTFMove(blobURLs))
    , m_blobFilePaths(WTFMove(blobFilePaths))
{
}

IDBValue::IDBValue(const ThreadSafeDataBuffer& value, const Vector<String>& blobURLs, const Vector<String>& blobFilePaths)
    : m_data(value)
    , m_blobURLs(blobURLs)
    , m_blobFilePaths(blobFilePaths)
{
}

void IDBValue::setAsIsolatedCopy(const IDBValue& other)
{
    ASSERT(m_blobURLs.isEmpty() && m_blobFilePaths.isEmpty());

    m_data = other.m_data;
    m_blobURLs = crossThreadCopy(other.m_blobURLs);
    m_blobFilePaths = crossThreadCopy(other.m_blobFilePaths);
}

IDBValue IDBValue::isolatedCopy() const
{
    IDBValue result;
    result.setAsIsolatedCopy(*this);
    return result;
}

size_t IDBValue::size() const
{
    size_t totalSize = 0;

    for (auto& url : m_blobURLs)
        totalSize += url.sizeInBytes();

    for (auto& path : m_blobFilePaths)
        totalSize += path.sizeInBytes();

    totalSize += m_data.size();

    return totalSize;
}

} // namespace WebCore
