/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#include "WebExtensionSQLiteRow.h"
#include "WebExtensionSQLiteStore.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

enum class WebExtensionDataType : uint8_t;

class WebExtensionStorageSQLiteStore final : public WebExtensionSQLiteStore {
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionStorageSQLiteStore);

public:
    template<typename... Args>
    static Ref<WebExtensionStorageSQLiteStore> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionStorageSQLiteStore(std::forward<Args>(args)...));
    }
    virtual ~WebExtensionStorageSQLiteStore() = default;

    void getAllKeys(CompletionHandler<void(Vector<String> keys, const String& errorMessage)>&&);
    void getValuesForKeys(Vector<String> keys, CompletionHandler<void(HashMap<String, String> results, const String& errorMessage)>&&);
    void getStorageSizeForKeys(Vector<String> keys, CompletionHandler<void(size_t storageSize, const String& errorMessage)>&&);
    void getStorageSizeForAllKeys(HashMap<String, String> additionalKeyedData, CompletionHandler<void(size_t storageSize, int numberOfKeysIncludingAdditionalKeyedData, HashMap<String, String> existingKeysAndValues, const String& errorMessage)>&&);
    void setKeyedData(HashMap<String, String> keyedData, CompletionHandler<void(Vector<String> keysSuccessfullySet, const String& errorMessage)>&&);
    void deleteValuesForKeys(Vector<String> keys, CompletionHandler<void(const String& errorMessage)>&&);

    enum class UsesInMemoryDatabase : bool {
        No = false,
        Yes = true,
    };

protected:
    SchemaVersion migrateToCurrentSchemaVersionIfNeeded();

    DatabaseResult createFreshDatabaseSchema() override;
    DatabaseResult resetDatabaseSchema() override;
    bool isDatabaseEmpty() override;
    SchemaVersion currentDatabaseSchemaVersion() override;
    URL databaseURL() override;

private:
    WebExtensionStorageSQLiteStore(const String& uniqueIdentifier, WebExtensionDataType storageType, const String& directory, UsesInMemoryDatabase useInMemoryDatabase);

    String insertOrUpdateValue(const String& value, const String& key, Ref<WebExtensionSQLiteDatabase>);
    HashMap<String, String> getValuesForAllKeys(String& errorMessage);
    HashMap<String, String> getValuesForKeysWithErrorMessage(Vector<String> keys, String& errorMessage);
    HashMap<String, String> getKeysAndValuesFromRowIterator(Ref<WebExtensionSQLiteRowEnumerator> rows);
    Vector<String> getAllKeysWithErrorMessage(String& errorMessage);

    WebExtensionDataType m_storageType;
};

} // namespace WebKit
