/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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
#include "IDBCursorInfo.h"

#include "IDBDatabase.h"
#include "IDBTransaction.h"
#include "IndexedDB.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

IDBCursorInfo IDBCursorInfo::objectStoreCursor(IDBTransaction& transaction, IDBObjectStoreIdentifier objectStoreIdentifier, const IDBKeyRangeData& range, IndexedDB::CursorDirection direction, IndexedDB::CursorType type)
{
    return { transaction, objectStoreIdentifier, range, direction, type };
}

IDBCursorInfo IDBCursorInfo::indexCursor(IDBTransaction& transaction, IDBObjectStoreIdentifier objectStoreIdentifier, IDBIndexIdentifier indexIdentifier, const IDBKeyRangeData& range, IndexedDB::CursorDirection direction, IndexedDB::CursorType type)
{
    return { transaction, objectStoreIdentifier, indexIdentifier, range, direction, type };
}

IDBCursorInfo::IDBCursorInfo(IDBTransaction& transaction, IDBObjectStoreIdentifier objectStoreIdentifier, const IDBKeyRangeData& range, IndexedDB::CursorDirection direction, IndexedDB::CursorType type)
    : m_cursorIdentifier(transaction.database().connectionProxy())
    , m_transactionIdentifier(transaction.info().identifier())
    , m_objectStoreIdentifier(objectStoreIdentifier)
    , m_sourceIdentifier(objectStoreIdentifier)
    , m_range(range)
    , m_source(IndexedDB::CursorSource::ObjectStore)
    , m_direction(direction)
    , m_type(type)
{
}

IDBCursorInfo::IDBCursorInfo(IDBTransaction& transaction, IDBObjectStoreIdentifier objectStoreIdentifier, IDBIndexIdentifier indexIdentifier, const IDBKeyRangeData& range, IndexedDB::CursorDirection direction, IndexedDB::CursorType type)
    : m_cursorIdentifier(transaction.database().connectionProxy())
    , m_transactionIdentifier(transaction.info().identifier())
    , m_objectStoreIdentifier(objectStoreIdentifier)
    , m_sourceIdentifier(indexIdentifier)
    , m_range(range)
    , m_source(IndexedDB::CursorSource::Index)
    , m_direction(direction)
    , m_type(type)
{
}

IDBCursorInfo::IDBCursorInfo(const IDBResourceIdentifier& cursorIdentifier, const IDBResourceIdentifier& transactionIdentifier, IDBObjectStoreIdentifier objectStoreIdentifier, std::variant<IDBObjectStoreIdentifier, IDBIndexIdentifier> sourceIdentifier, const IDBKeyRangeData& range, IndexedDB::CursorSource source, IndexedDB::CursorDirection direction, IndexedDB::CursorType type)
    : m_cursorIdentifier(cursorIdentifier)
    , m_transactionIdentifier(transactionIdentifier)
    , m_objectStoreIdentifier(objectStoreIdentifier)
    , m_sourceIdentifier(sourceIdentifier)
    , m_range(range)
    , m_source(source)
    , m_direction(direction)
    , m_type(type)
{
}

bool IDBCursorInfo::isDirectionForward() const
{
    return m_direction == IndexedDB::CursorDirection::Next || m_direction == IndexedDB::CursorDirection::Nextunique;
}

CursorDuplicity IDBCursorInfo::duplicity() const
{
    return m_direction == IndexedDB::CursorDirection::Nextunique || m_direction == IndexedDB::CursorDirection::Prevunique ? CursorDuplicity::NoDuplicates : CursorDuplicity::Duplicates;
}

IDBCursorInfo IDBCursorInfo::isolatedCopy() const
{
    return { m_cursorIdentifier.isolatedCopy(), m_transactionIdentifier.isolatedCopy(), m_objectStoreIdentifier, m_sourceIdentifier, m_range.isolatedCopy(), m_source, m_direction, m_type };
}

std::optional<IDBIndexIdentifier> IDBCursorInfo::sourceIndexIdentifier() const
{
    if (m_source == IndexedDB::CursorSource::Index)
        return std::get<IDBIndexIdentifier>(m_sourceIdentifier);
    return std::nullopt;
}

#if !LOG_DISABLED

String IDBCursorInfo::loggingString() const
{
    auto sourceIdentifier = std::holds_alternative<IDBObjectStoreIdentifier>(m_sourceIdentifier) ? std::get<IDBObjectStoreIdentifier>(m_sourceIdentifier).toRawValue() : std::get<IDBIndexIdentifier>(m_sourceIdentifier).toRawValue();

    if (m_source == IndexedDB::CursorSource::Index)
        return makeString("<Crsr: "_s, m_cursorIdentifier.loggingString(), " Idx "_s, sourceIdentifier, ", OS "_s, m_objectStoreIdentifier, ", tx "_s, m_transactionIdentifier.loggingString(), '>');

    return makeString("<Crsr: "_s, m_cursorIdentifier.loggingString(), " OS "_s, m_objectStoreIdentifier, ", tx "_s, m_transactionIdentifier.loggingString(), '>');
}

#endif

} // namespace WebCore
