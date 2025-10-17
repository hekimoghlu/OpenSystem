/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
#include "IndexValueEntry.h"

#include "IDBCursorInfo.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
namespace IDBServer {

WTF_MAKE_TZONE_ALLOCATED_IMPL(IndexValueEntry);

IndexValueEntry::IndexValueEntry(bool unique)
    : m_unique(unique)
{
    if (m_unique)
        m_key = nullptr;
    else
        m_orderedKeys = new IDBKeyDataSet;
}

IndexValueEntry::~IndexValueEntry()
{
    if (m_unique)
        delete m_key;
    else
        delete m_orderedKeys;
}

void IndexValueEntry::addKey(const IDBKeyData& key)
{
    if (m_unique) {
        delete m_key;
        m_key = new IDBKeyData(key);
        return;
    }

    m_orderedKeys->insert(key);
}

bool IndexValueEntry::removeKey(const IDBKeyData& key)
{
    if (m_unique) {
        if (m_key && *m_key == key) {
            delete m_key;
            m_key = nullptr;
            return true;
        }

        return false;
    }

    return m_orderedKeys->erase(key);
}

const IDBKeyData* IndexValueEntry::getLowest() const
{
    if (m_unique)
        return m_key;

    if (m_orderedKeys->empty())
        return nullptr;

    return &(*m_orderedKeys->begin());
}

uint64_t IndexValueEntry::getCount() const
{
    if (m_unique)
        return m_key ? 1 : 0;

    return m_orderedKeys->size();
}

IndexValueEntry::Iterator::Iterator(IndexValueEntry& entry)
    : m_entry(&entry)
{
    ASSERT(m_entry->m_key);
}

IndexValueEntry::Iterator::Iterator(IndexValueEntry& entry, IDBKeyDataSet::iterator iterator)
    : m_entry(&entry)
    , m_forwardIterator(iterator)
{
}

IndexValueEntry::Iterator::Iterator(IndexValueEntry& entry, IDBKeyDataSet::reverse_iterator iterator)
    : m_entry(&entry)
    , m_forward(false)
    , m_reverseIterator(iterator)
{
}

const IDBKeyData& IndexValueEntry::Iterator::key() const
{
    ASSERT(isValid());
    if (m_entry->unique()) {
        ASSERT(m_entry->m_key);
        return *m_entry->m_key;
    }

    return m_forward ? *m_forwardIterator : *m_reverseIterator;
}

bool IndexValueEntry::Iterator::isValid() const
{
#if !LOG_DISABLED
    if (m_entry) {
        if (m_entry->m_unique)
            ASSERT(m_entry->m_key);
        else
            ASSERT(m_entry->m_orderedKeys);
    }
#endif

    return m_entry;
}

void IndexValueEntry::Iterator::invalidate()
{
    m_entry = nullptr;
}

IndexValueEntry::Iterator& IndexValueEntry::Iterator::operator++()
{
    if (!isValid())
        return *this;

    if (m_entry->m_unique) {
        invalidate();
        return *this;
    }

    if (m_forward) {
        ++m_forwardIterator;
        if (m_forwardIterator == m_entry->m_orderedKeys->end())
            invalidate();
    } else {
        ++m_reverseIterator;
        if (m_reverseIterator == m_entry->m_orderedKeys->rend())
            invalidate();
    }

    return *this;
}

IndexValueEntry::Iterator IndexValueEntry::begin()
{
    if (m_unique) {
        ASSERT(m_key);
        return { *this };
    }

    ASSERT(m_orderedKeys);
    return { *this, m_orderedKeys->begin() };
}

IndexValueEntry::Iterator IndexValueEntry::reverseBegin(CursorDuplicity duplicity)
{
    if (m_unique) {
        ASSERT(m_key);
        return { *this };
    }

    ASSERT(m_orderedKeys);

    if (duplicity == CursorDuplicity::Duplicates)
        return { *this, m_orderedKeys->rbegin() };

    auto iterator = m_orderedKeys->rend();
    --iterator;
    return { *this, iterator };
}

IndexValueEntry::Iterator IndexValueEntry::find(const IDBKeyData& key)
{
    if (m_unique) {
        ASSERT(m_key);
        return *m_key == key ? IndexValueEntry::Iterator(*this) : IndexValueEntry::Iterator();
    }

    ASSERT(m_orderedKeys);
    auto iterator = m_orderedKeys->lower_bound(key);
    if (iterator == m_orderedKeys->end())
        return { };

    return { *this, iterator };
}

IndexValueEntry::Iterator IndexValueEntry::reverseFind(const IDBKeyData& key, CursorDuplicity)
{
    if (m_unique) {
        ASSERT(m_key);
        return *m_key == key ? IndexValueEntry::Iterator(*this) : IndexValueEntry::Iterator();
    }

    ASSERT(m_orderedKeys);
    auto iterator = IDBKeyDataSet::reverse_iterator(m_orderedKeys->upper_bound(key));
    if (iterator == m_orderedKeys->rend())
        return { };

    return { *this, iterator };
}


} // namespace IDBServer
} // namespace WebCore
