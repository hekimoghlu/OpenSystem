/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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

#include <wtf/Deque.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class SQLTransaction;

class SQLTransactionCoordinator {
    WTF_MAKE_TZONE_ALLOCATED(SQLTransactionCoordinator);
    WTF_MAKE_NONCOPYABLE(SQLTransactionCoordinator);
public:
    SQLTransactionCoordinator();
    void acquireLock(SQLTransaction&);
    void releaseLock(SQLTransaction&);
    void shutdown();
private:
    typedef Deque<RefPtr<SQLTransaction>> TransactionsQueue;
    struct CoordinationInfo {
        TransactionsQueue pendingTransactions;
        HashSet<RefPtr<SQLTransaction>> activeReadTransactions;
        RefPtr<SQLTransaction> activeWriteTransaction;
    };
    // Maps database names to information about pending transactions
    typedef HashMap<String, CoordinationInfo> CoordinationInfoMap;
    CoordinationInfoMap m_coordinationInfoMap;
    bool m_isShuttingDown;

    void processPendingTransactions(CoordinationInfo&);
};

} // namespace WebCore
