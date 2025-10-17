/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 2, 2022.
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
#include "IDBKeyPath.h"
#include "IDBValue.h"
#include "IndexedDB.h"
#include <variant>
#include <wtf/ArgumentCoder.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class IDBGetAllResult {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(IDBGetAllResult, WEBCORE_EXPORT);
public:
    IDBGetAllResult() = default;

    IDBGetAllResult(IndexedDB::GetAllType type, const std::optional<IDBKeyPath>& keyPath)
        : m_type(type)
        , m_keyPath(keyPath)
    {
    }

    enum IsolatedCopyTag { IsolatedCopy };
    IDBGetAllResult(const IDBGetAllResult&, IsolatedCopyTag);
    IDBGetAllResult isolatedCopy() const;

    IndexedDB::GetAllType type() const { return m_type; }
    const std::optional<IDBKeyPath>& keyPath() const { return m_keyPath; }
    WEBCORE_EXPORT const Vector<IDBKeyData>& keys() const;
    WEBCORE_EXPORT const Vector<IDBValue>& values() const;

    void addKey(IDBKeyData&&);
    void addValue(IDBValue&&);

    WEBCORE_EXPORT Vector<String> allBlobFilePaths() const;

private:
    friend struct IPC::ArgumentCoder<IDBGetAllResult, void>;
    IDBGetAllResult(IndexedDB::GetAllType type, Vector<IDBKeyData>&& keys, Vector<IDBValue>&& values, std::optional<IDBKeyPath>&& keyPath)
        : m_type(type)
        , m_keys(WTFMove(keys))
        , m_values(WTFMove(values))
        , m_keyPath(WTFMove(keyPath))
    {
    }

    static void isolatedCopy(const IDBGetAllResult& source, IDBGetAllResult& destination);

    IndexedDB::GetAllType m_type { IndexedDB::GetAllType::Keys };
    Vector<IDBKeyData> m_keys;
    Vector<IDBValue> m_values;
    std::optional<IDBKeyPath> m_keyPath;
};

} // namespace WebCore
