/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 12, 2023.
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

#include "IDBCursorInfo.h"
#include "IDBKeyData.h"
#include "MemoryCursor.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
namespace IDBServer {

class MemoryBackingStoreTransaction;
class MemoryObjectStore;

class MemoryObjectStoreCursor : public MemoryCursor {
    WTF_MAKE_TZONE_ALLOCATED(MemoryObjectStoreCursor);
public:
    static Ref<MemoryObjectStoreCursor> create(MemoryObjectStore&, const IDBCursorInfo&, MemoryBackingStoreTransaction&);

    void objectStoreCleared();
    void keyDeleted(const IDBKeyData&);
    void keyAdded(IDBKeyDataSet::iterator);

private:
    MemoryObjectStoreCursor(MemoryObjectStore&, const IDBCursorInfo&, MemoryBackingStoreTransaction&);
    void currentData(IDBGetResult&) final;
    void iterate(const IDBKeyData&, const IDBKeyData& primaryKey, uint32_t count, IDBGetResult&) final;

    void setFirstInRemainingRange(IDBKeyDataSet&);
    void setForwardIteratorFromRemainingRange(IDBKeyDataSet&);
    void setReverseIteratorFromRemainingRange(IDBKeyDataSet&);

    void incrementForwardIterator(IDBKeyDataSet&, const IDBKeyData&, uint32_t count);
    void incrementReverseIterator(IDBKeyDataSet&, const IDBKeyData&, uint32_t count);

    bool hasValidPosition() const;

    WeakRef<MemoryObjectStore> m_objectStore;

    IDBKeyRangeData m_remainingRange;

    std::optional<IDBKeyDataSet::iterator> m_iterator;

    IDBKeyData m_currentPositionKey;
};

} // namespace IDBServer
} // namespace WebCore
