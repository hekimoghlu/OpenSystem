/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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

#include "StorageAreaBase.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class SQLiteDatabase;
class SQLiteStatement;
class SQLiteStatementAutoResetScope;
class SQLiteTransaction;
}

namespace WebKit {

class SQLiteStorageArea final : public StorageAreaBase, public RefCounted<SQLiteStorageArea> {
    WTF_MAKE_TZONE_ALLOCATED(SQLiteStorageArea);
public:
    static Ref<SQLiteStorageArea> create(unsigned quota, const WebCore::ClientOrigin&, const String& path, Ref<WorkQueue>&&);
    ~SQLiteStorageArea();

    void close();
    void handleLowMemoryWarning();
    void commitTransactionIfNecessary();
    void clear() final;

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    SQLiteStorageArea(unsigned quota, const WebCore::ClientOrigin&, const String& path, Ref<WorkQueue>&&);

    // StorageAreaBase
    Type type() const final { return StorageAreaBase::Type::SQLite; };
    StorageType storageType() const final { return StorageAreaBase::StorageType::Local; };
    bool isEmpty() final;
    HashMap<String, String> allItems() final;
    Expected<void, StorageError> setItem(std::optional<IPC::Connection::UniqueID>, std::optional<StorageAreaImplIdentifier>, String&& key, String&& value, const String& urlString) final;
    Expected<void, StorageError> removeItem(IPC::Connection::UniqueID, StorageAreaImplIdentifier, const String& key, const String& urlString) final;
    Expected<void, StorageError> clear(IPC::Connection::UniqueID, StorageAreaImplIdentifier, const String& urlString) final;

    bool createTableIfNecessary();
    enum class ShouldCreateIfNotExists : bool { No, Yes };
    bool prepareDatabase(ShouldCreateIfNotExists);
    void startTransactionIfNecessary();

    enum class StatementType : uint8_t {
        CountItems,
        DeleteItem,
        DeleteAllItems,
        GetItem,
        GetAllItems,
        SetItem,
        Invalid
    };
    ASCIILiteral statementString(StatementType) const;
    WebCore::SQLiteStatementAutoResetScope cachedStatement(StatementType);
    Expected<String, StorageError> getItem(const String& key);
    Expected<String, StorageError> getItemFromDatabase(const String& key);
    enum class IsDatabaseDeleted : bool { No, Yes };
    IsDatabaseDeleted handleDatabaseErrorIfNeeded(int databaseError);
    void updateCacheIfNeeded(const String& key, const String& value);
    bool requestSpace(const String& key, const String& value);

    String m_path;
    Ref<WorkQueue> m_queue;
    std::unique_ptr<WebCore::SQLiteDatabase> m_database;
    std::unique_ptr<WebCore::SQLiteTransaction> m_transaction;
    Vector<std::unique_ptr<WebCore::SQLiteStatement>> m_cachedStatements;
    using Value = std::variant<String, unsigned>;
    std::optional<HashMap<String, Value>> m_cache;
    std::optional<unsigned> m_cacheSize;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::SQLiteStorageArea)
    static bool isType(const WebKit::StorageAreaBase& area) { return area.type() == WebKit::StorageAreaBase::Type::SQLite; }
SPECIALIZE_TYPE_TRAITS_END()
