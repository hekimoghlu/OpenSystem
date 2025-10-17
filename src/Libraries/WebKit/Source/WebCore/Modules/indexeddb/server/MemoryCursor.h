/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebCore {

class IDBGetResult;
class IDBKeyData;
class IDBResourceIdentifier;

namespace IDBServer {

class MemoryBackingStoreTransaction;

class MemoryCursor : public RefCountedAndCanMakeWeakPtr<MemoryCursor> {
    WTF_MAKE_TZONE_ALLOCATED(MemoryCursor);
public:
    virtual ~MemoryCursor();

    virtual void currentData(IDBGetResult&) = 0;
    virtual void iterate(const IDBKeyData&, const IDBKeyData& primaryKey, uint32_t count, IDBGetResult&) = 0;

    MemoryBackingStoreTransaction* transaction() const { return m_transaction.get(); }
    IDBCursorInfo info() const { return m_info; }

protected:
    MemoryCursor(const IDBCursorInfo&, MemoryBackingStoreTransaction&);

private:
    IDBCursorInfo m_info;
    WeakPtr<MemoryBackingStoreTransaction> m_transaction;
};

} // namespace IDBServer
} // namespace WebCore
