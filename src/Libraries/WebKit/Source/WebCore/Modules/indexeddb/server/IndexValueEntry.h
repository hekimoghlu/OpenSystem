/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ThreadSafeDataBuffer;

enum class CursorDuplicity;

namespace IDBServer {

class IndexValueEntry {
    WTF_MAKE_TZONE_ALLOCATED(IndexValueEntry);
public:
    explicit IndexValueEntry(bool unique);
    ~IndexValueEntry();

    void addKey(const IDBKeyData&);

    // Returns true if a key was actually removed.
    bool removeKey(const IDBKeyData&);

    const IDBKeyData* getLowest() const;

    uint64_t getCount() const;

    class Iterator {
    public:
        Iterator()
        {
        }

        Iterator(IndexValueEntry&);
        Iterator(IndexValueEntry&, IDBKeyDataSet::iterator);
        Iterator(IndexValueEntry&, IDBKeyDataSet::reverse_iterator);

        bool isValid() const;
        void invalidate();

        const IDBKeyData& key() const;
        const ThreadSafeDataBuffer& value() const;

        Iterator& operator++();

    private:
        IndexValueEntry* m_entry { nullptr };
        bool m_forward { true };
        IDBKeyDataSet::iterator m_forwardIterator;
        IDBKeyDataSet::reverse_iterator m_reverseIterator;
    };

    Iterator begin();
    Iterator reverseBegin(CursorDuplicity);

    // Finds the key, or the next higher record after the key.
    Iterator find(const IDBKeyData&);
    // Finds the key, or the next lowest record before the key.
    Iterator reverseFind(const IDBKeyData&, CursorDuplicity);

    bool unique() const { return m_unique; }

private:
    union {
        IDBKeyDataSet* m_orderedKeys;
        IDBKeyData* m_key;
    };

    bool m_unique;
};

} // namespace IDBServer
} // namespace WebCore
