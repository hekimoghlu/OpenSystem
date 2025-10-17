/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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

#include "IDBDatabaseIdentifier.h"
#include <wtf/WeakPtr.h>

namespace WebCore {
namespace IDBServer {
class UniqueIDBDatabaseManager;
}
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::IDBServer::UniqueIDBDatabaseManager> : std::true_type { };
}

namespace WebCore {

struct ClientOrigin;

namespace IDBServer {

class IDBBackingStore;
class UniqueIDBDatabaseConnection;
class UniqueIDBDatabaseTransaction;

class UniqueIDBDatabaseManager : public CanMakeWeakPtr<UniqueIDBDatabaseManager> {
    WTF_MAKE_TZONE_NON_HEAP_ALLOCATABLE(UniqueIDBDatabaseManager);
public:
    virtual ~UniqueIDBDatabaseManager() { }
    virtual void registerConnection(UniqueIDBDatabaseConnection&) = 0;
    virtual void unregisterConnection(UniqueIDBDatabaseConnection&) = 0;
    virtual void registerTransaction(UniqueIDBDatabaseTransaction&) = 0;
    virtual void unregisterTransaction(UniqueIDBDatabaseTransaction&) = 0;
    virtual std::unique_ptr<IDBBackingStore> createBackingStore(const IDBDatabaseIdentifier&) = 0;
    virtual void requestSpace(const ClientOrigin&, uint64_t size, CompletionHandler<void(bool)>&&) = 0;
};

} // namespace IDBServer

} // namespace WebCore
