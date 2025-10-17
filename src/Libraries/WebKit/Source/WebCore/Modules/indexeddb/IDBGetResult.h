/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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

#include "IDBCursorRecord.h"
#include "IDBKey.h"
#include "IDBKeyData.h"
#include "IDBKeyPath.h"
#include "IDBValue.h"
#include "SharedBuffer.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class IDBGetResult {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(IDBGetResult, WEBCORE_EXPORT);
public:
    IDBGetResult()
        : m_isDefined(false)
    {
    }

    IDBGetResult(const IDBKeyData& keyData)
        : m_keyData(keyData)
    {
    }

    IDBGetResult(const IDBKeyData& keyData, const IDBKeyData& primaryKeyData)
        : m_keyData(keyData)
        , m_primaryKeyData(primaryKeyData)
    {
    }

    IDBGetResult(const IDBKeyData& keyData, const ThreadSafeDataBuffer& buffer, const std::optional<IDBKeyPath>& keyPath)
        : m_value(buffer)
        , m_keyData(keyData)
        , m_keyPath(keyPath)
    {
    }

    IDBGetResult(const IDBKeyData& keyData, IDBValue&& value, const std::optional<IDBKeyPath>& keyPath)
        : m_value(WTFMove(value))
        , m_keyData(keyData)
        , m_keyPath(keyPath)
    {
    }

    IDBGetResult(const IDBKeyData& keyData, const IDBKeyData& primaryKeyData, IDBValue&& value, const std::optional<IDBKeyPath>& keyPath, Vector<IDBCursorRecord>&& prefetechedRecords = { }, bool isDefined = true)
        : m_value(WTFMove(value))
        , m_keyData(keyData)
        , m_primaryKeyData(primaryKeyData)
        , m_keyPath(keyPath)
        , m_prefetchedRecords(WTFMove(prefetechedRecords))
        , m_isDefined(isDefined)
    {
    }

    enum IsolatedCopyTag { IsolatedCopy };
    IDBGetResult(const IDBGetResult&, IsolatedCopyTag);

    IDBGetResult isolatedCopy() const;

    void setValue(IDBValue&&);

    const IDBValue& value() const { return m_value; }
    const IDBKeyData& keyData() const { return m_keyData; }
    const IDBKeyData& primaryKeyData() const { return m_primaryKeyData; }
    const std::optional<IDBKeyPath>& keyPath() const { return m_keyPath; }
    const Vector<IDBCursorRecord>& prefetchedRecords() const { return m_prefetchedRecords; }
    bool isDefined() const { return m_isDefined; }

private:
    static void isolatedCopy(const IDBGetResult& source, IDBGetResult& destination);

    IDBValue m_value;
    IDBKeyData m_keyData;
    IDBKeyData m_primaryKeyData;
    std::optional<IDBKeyPath> m_keyPath;
    Vector<IDBCursorRecord> m_prefetchedRecords;
    bool m_isDefined { true };
};

} // namespace WebCore
