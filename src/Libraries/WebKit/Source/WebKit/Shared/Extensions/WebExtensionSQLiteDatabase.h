/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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

#include "APIError.h"
#include <sqlite3.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/URL.h>
#include <wtf/WorkQueue.h>

struct sqlite3;

namespace WebKit {

class WebExtensionSQLiteStatement;
class WebExtensionSQLiteStore;

class WebExtensionSQLiteDatabase final : public RefCounted<WebExtensionSQLiteDatabase> {
    WTF_MAKE_NONCOPYABLE(WebExtensionSQLiteDatabase);
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionSQLiteDatabase);

    friend class WebExtensionSQLiteStatement;
    friend class WebExtensionSQLiteStore;

public:
    template<typename... Args>
    static Ref<WebExtensionSQLiteDatabase> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionSQLiteDatabase(std::forward<Args>(args)...));
    }

    explicit WebExtensionSQLiteDatabase(const URL&, Ref<WorkQueue>&&);
    ~WebExtensionSQLiteDatabase()
    {
        ASSERT(!m_db);
    }

    static URL inMemoryDatabaseURL();

    enum class AccessType : uint8_t {
        ReadOnly = 0,
        ReadWrite,
        ReadWriteCreate
    };

    // This enum is only applicable on iOS and has no effect on other platforms.
    // ProtectionType::Default sets the protection to class C.
    enum class ProtectionType : uint8_t {
        Default = 0,
        CompleteUntilFirstUserAuthentication,
        CompleteUnlessOpen,
        Complete
    };

    bool openWithAccessType(AccessType, RefPtr<API::Error>&, ProtectionType = { }, const String& vfs = { });
    bool enableWAL(RefPtr<API::Error>&);

    void reportErrorWithCode(int, const String& query, RefPtr<API::Error>&);
    void reportErrorWithCode(int, sqlite3_stmt* statement, RefPtr<API::Error>&);

    int close();

    sqlite3* sqlite3Handle() const { return m_db; };
    void assertQueue();
    Ref<WorkQueue> queue() const { return m_queue; };

private:
    RefPtr<API::Error> errorWithSQLiteErrorCode(int errorCode);
    URL privateOnDiskDatabaseURL();

    sqlite3* m_db { nullptr };
    URL m_url;

    int m_lastErrorCode;
    CString m_lastErrorMessage;

    Ref<WorkQueue> m_queue;
};

}; // namespace WebKit
