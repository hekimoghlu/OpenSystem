/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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

#include "SharedStringHashTable.h"
#include <WebCore/SharedMemory.h>
#include <WebCore/SharedStringHash.h>
#include <wtf/CheckedPtr.h>
#include <wtf/RunLoop.h>

namespace WebKit {

class SharedStringHashStore : public CanMakeCheckedPtr<SharedStringHashStore> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SharedStringHashStore);
public:
    class Client {
    public:
        virtual ~Client() { }

        virtual void didInvalidateSharedMemory() = 0;
        virtual void didUpdateSharedStringHashes(const Vector<WebCore::SharedStringHash>& addedHashes, const Vector<WebCore::SharedStringHash>& removedHashes) { };
    };

    SharedStringHashStore(Client&);

    std::optional<WebCore::SharedMemory::Handle> createSharedMemoryHandle();

    void scheduleAddition(WebCore::SharedStringHash);
    void scheduleRemoval(WebCore::SharedStringHash);

    bool contains(WebCore::SharedStringHash);
    void clear();

    bool isEmpty() const { return !m_keyCount; }

    void flushPendingChanges();

private:
    void resizeTable(unsigned newTableLength);
    void processPendingOperations();

    struct Operation {
        enum Type { Add, Remove };
        Type type;
        WebCore::SharedStringHash sharedStringHash;
    };

    Client& m_client;
    unsigned m_keyCount { 0 };
    unsigned m_tableLength { 0 };
    SharedStringHashTable m_table;
    Vector<Operation> m_pendingOperations;
    RunLoop::Timer m_pendingOperationsTimer;
};

} // namespace WebKit
