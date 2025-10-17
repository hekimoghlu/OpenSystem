/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

#include "IDBConnectionToClient.h"
#include "IDBDatabaseConnectionIdentifier.h"
#include "IDBOpenRequestData.h"
#include <wtf/HashSet.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class IDBDatabaseInfo;

namespace IDBServer {

class ServerOpenDBRequest : public RefCounted<ServerOpenDBRequest> {
public:
    static Ref<ServerOpenDBRequest> create(IDBConnectionToClient&, const IDBOpenRequestData&);

    IDBConnectionToClient& connection() { return m_connection; }
    const IDBOpenRequestData& requestData() const { return m_requestData; }

    bool isOpenRequest() const;
    bool isDeleteRequest() const;

    void maybeNotifyRequestBlocked(uint64_t currentVersion);
    void notifyDidDeleteDatabase(const IDBDatabaseInfo&);

    uint64_t versionChangeID() const;

    void notifiedConnectionsOfVersionChange(HashSet<IDBDatabaseConnectionIdentifier>&& connectionIdentifiers);
    void connectionClosedOrFiredVersionChangeEvent(IDBDatabaseConnectionIdentifier);
    bool hasConnectionsPendingVersionChangeEvent() const { return !m_connectionsPendingVersionChangeEvent.isEmpty(); }
    bool hasNotifiedConnectionsOfVersionChange() const { return m_notifiedConnectionsOfVersionChange; }


private:
    ServerOpenDBRequest(IDBConnectionToClient&, const IDBOpenRequestData&);

    Ref<IDBConnectionToClient> m_connection;
    IDBOpenRequestData m_requestData;

    bool m_notifiedBlocked { false };

    bool m_notifiedConnectionsOfVersionChange { false };
    HashSet<IDBDatabaseConnectionIdentifier> m_connectionsPendingVersionChangeEvent;
};

} // namespace IDBServer
} // namespace WebCore
