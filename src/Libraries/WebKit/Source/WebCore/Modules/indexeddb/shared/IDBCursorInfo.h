/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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

#include "IDBIndexIdentifier.h"
#include "IDBKeyRangeData.h"
#include "IDBObjectStoreIdentifier.h"
#include "IDBResourceIdentifier.h"
#include <wtf/ArgumentCoder.h>

namespace WebCore {

class IDBTransaction;

namespace IndexedDB {
enum class CursorDirection : uint8_t;
enum class CursorSource : bool;
enum class CursorType : bool;
}

struct IDBKeyRangeData;

enum class CursorDuplicity {
    Duplicates,
    NoDuplicates,
};

class IDBCursorInfo {
public:
    static IDBCursorInfo objectStoreCursor(IDBTransaction&, IDBObjectStoreIdentifier, const IDBKeyRangeData&, IndexedDB::CursorDirection, IndexedDB::CursorType);
    static IDBCursorInfo indexCursor(IDBTransaction&, IDBObjectStoreIdentifier, IDBIndexIdentifier, const IDBKeyRangeData&, IndexedDB::CursorDirection, IndexedDB::CursorType);

    IDBResourceIdentifier identifier() const { return m_cursorIdentifier; }
    IDBResourceIdentifier transactionIdentifier() const { return m_transactionIdentifier; }
    std::variant<IDBObjectStoreIdentifier, IDBIndexIdentifier> sourceIdentifier() const { return m_sourceIdentifier; }
    std::optional<IDBIndexIdentifier> sourceIndexIdentifier() const;
    IDBObjectStoreIdentifier objectStoreIdentifier() const { return m_objectStoreIdentifier; }

    IndexedDB::CursorSource cursorSource() const { return m_source; }
    IndexedDB::CursorDirection cursorDirection() const { return m_direction; }
    IndexedDB::CursorType cursorType() const { return m_type; }
    const IDBKeyRangeData& range() const { return m_range; }

    bool isDirectionForward() const;
    CursorDuplicity duplicity() const;

    WEBCORE_EXPORT IDBCursorInfo isolatedCopy() const;

#if !LOG_DISABLED
    String loggingString() const;
#endif

private:
    friend struct IPC::ArgumentCoder<IDBCursorInfo, void>;
    WEBCORE_EXPORT IDBCursorInfo(const IDBResourceIdentifier&, const IDBResourceIdentifier&, IDBObjectStoreIdentifier, std::variant<IDBObjectStoreIdentifier, IDBIndexIdentifier>, const IDBKeyRangeData&, IndexedDB::CursorSource, IndexedDB::CursorDirection, IndexedDB::CursorType);
    IDBCursorInfo(IDBTransaction&, IDBObjectStoreIdentifier, const IDBKeyRangeData&, IndexedDB::CursorDirection, IndexedDB::CursorType);
    IDBCursorInfo(IDBTransaction&, IDBObjectStoreIdentifier, IDBIndexIdentifier, const IDBKeyRangeData&, IndexedDB::CursorDirection, IndexedDB::CursorType);

    IDBResourceIdentifier m_cursorIdentifier;
    IDBResourceIdentifier m_transactionIdentifier;
    IDBObjectStoreIdentifier m_objectStoreIdentifier;
    std::variant<IDBObjectStoreIdentifier, IDBIndexIdentifier> m_sourceIdentifier;

    IDBKeyRangeData m_range;

    IndexedDB::CursorSource m_source;
    IndexedDB::CursorDirection m_direction;
    IndexedDB::CursorType m_type;
};

} // namespace WebCore
