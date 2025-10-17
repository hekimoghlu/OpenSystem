/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 14, 2024.
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

#include "StorageConnection.h"
#include <wtf/WeakPtr.h>

namespace WebCore {

class WeakPtrImplWithEventTargetData;
class WorkerGlobalScope;
struct StorageEstimate;

class WorkerStorageConnection final : public StorageConnection  {
public:
    static Ref<WorkerStorageConnection> create(WorkerGlobalScope&);
    void scopeClosed();

private:
    explicit WorkerStorageConnection(WorkerGlobalScope&);
    void didGetPersisted(uint64_t callbackIdentifier, bool result);
    void didGetEstimate(uint64_t callbackIdentifier, ExceptionOr<StorageEstimate>&&);
    void didGetDirectory(uint64_t callbackIdentifier, ExceptionOr<StorageConnection::DirectoryInfo>&&);

    // StorageConnection
    void getPersisted(ClientOrigin&&, StorageConnection::PersistCallback&&) final;
    void getEstimate(ClientOrigin&&, StorageConnection::GetEstimateCallback&&) final;
    void fileSystemGetDirectory(ClientOrigin&&, StorageConnection::GetDirectoryCallback&&) final;

    WeakPtr<WorkerGlobalScope, WeakPtrImplWithEventTargetData> m_scope;
    uint64_t m_lastCallbackIdentifier { 0 };
    HashMap<uint64_t, StorageConnection::PersistCallback> m_getPersistedCallbacks;
    HashMap<uint64_t, StorageConnection::GetEstimateCallback> m_getEstimateCallbacks;
    HashMap<uint64_t, StorageConnection::GetDirectoryCallback> m_getDirectoryCallbacks;
};

} // namespace WebCore
