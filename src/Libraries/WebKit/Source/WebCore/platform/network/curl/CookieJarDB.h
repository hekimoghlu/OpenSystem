/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#include "Cookie.h"
#include "CookieJar.h"
#include "SQLiteDatabase.h"
#include "SQLiteStatement.h"
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

enum class CookieAcceptPolicy {
    Always,
    Never,
    OnlyFromMainDocumentDomain,
    ExclusivelyFromMainDocumentDomain
};

class CookieJarDB {
    WTF_MAKE_TZONE_ALLOCATED(CookieJarDB);
    WTF_MAKE_NONCOPYABLE(CookieJarDB);

public:
    enum class Source : uint8_t {
        Network,
        Script
    };

    WEBCORE_EXPORT void open();
    bool isEnabled() const;

    void setAcceptPolicy(CookieAcceptPolicy policy) { m_acceptPolicy = policy; }
    CookieAcceptPolicy acceptPolicy() const { return m_acceptPolicy; }

    HashSet<String> allDomains();
    std::optional<Vector<Cookie>> searchCookies(const URL& firstParty, const URL& requestUrl, const std::optional<bool>& httpOnly, const std::optional<bool>& secure, const std::optional<bool>& session);
    Vector<Cookie> getAllCookies();
    WEBCORE_EXPORT bool setCookie(const URL& firstParty, const URL&, const String& cookie, Source, std::optional<Seconds> cappedLifetime = std::nullopt);
    bool setCookie(const Cookie&);

    bool deleteCookie(const String& url, const String& name);
    bool deleteCookies(const String& url);
    bool deleteCookiesForHostname(const String& hostname, IncludeHttpOnlyCookies);
    bool deleteAllCookies();

    WEBCORE_EXPORT CookieJarDB(const String& databasePath);
    WEBCORE_EXPORT ~CookieJarDB();

private:
    CookieAcceptPolicy m_acceptPolicy { CookieAcceptPolicy::Always };
    String m_databasePath;

    bool m_detectedDatabaseCorruption { false };

    bool isOnMemory() const { return m_databasePath == ":memory:"_s; };

    bool openDatabase();
    void closeDatabase();

    bool checkSQLiteReturnCode(int);
    void flagDatabaseCorruption();
    bool checkDatabaseCorruptionAndRemoveIfNeeded();
    String getCorruptionMarkerPath() const;

    bool checkDatabaseValidity();
    void deleteAllDatabaseFiles();

    void verifySchemaVersion();
    void deleteAllTables();

    void createPrepareStatement(ASCIILiteral);
    SQLiteStatement& preparedStatement(const String&);
    bool executeSQLStatement(Expected<SQLiteStatement, int>&&);

    bool deleteCookieInternal(const String& name, const String& domain, const String& path);
    bool hasHttpOnlyCookie(const String& name, const String& domain, const String& path);
    bool canAcceptCookie(const Cookie&, const URL& firstParty, const URL&, CookieJarDB::Source);
    bool checkCookieAcceptPolicy(const URL& firstParty, const URL&);
    bool hasCookies(const URL&);

    SQLiteDatabase m_database;
    HashMap<String, std::unique_ptr<SQLiteStatement>> m_statements;
};

} // namespace WebCore
