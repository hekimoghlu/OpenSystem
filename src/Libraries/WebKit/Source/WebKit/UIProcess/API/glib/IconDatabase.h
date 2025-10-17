/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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

#include <WebCore/NativeImage.h>
#include <WebCore/SQLiteDatabase.h>
#include <WebCore/SQLiteStatement.h>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WorkQueue.h>

namespace WebKit {

class IconDatabase : public ThreadSafeRefCounted<IconDatabase> {
public:
    enum class AllowDatabaseWrite : bool { No, Yes };

    static Ref<IconDatabase> create(const String& path, AllowDatabaseWrite allowDatabaseWrite)
    {
        return adoptRef(*new IconDatabase(path, allowDatabaseWrite));
    }

    ~IconDatabase();

    void invalidate();

    void checkIconURLAndSetPageURLIfNeeded(const String& iconURL, const String& pageURL, AllowDatabaseWrite, CompletionHandler<void(bool, bool)>&&);
    void loadIconForPageURL(const String&, AllowDatabaseWrite, CompletionHandler<void(WebCore::PlatformImagePtr&&)>&&);
    String iconURLForPageURL(const String&);
    void setIconForPageURL(const String& iconURL, std::span<const uint8_t>, const String& pageURL, AllowDatabaseWrite, CompletionHandler<void(bool)>&&);
    void clear(CompletionHandler<void()>&&);

private:
    IconDatabase(const String&, AllowDatabaseWrite);

    bool createTablesIfNeeded();
    void populatePageURLToIconURLMap();
    void pruneTimerFired();
    void startPruneTimer();
    void clearStatements();
    void clearLoadedIconsTimerFired();
    void startClearLoadedIconsTimer();
    std::optional<int64_t> iconIDForIconURL(const String&, bool& expired);
    bool setIconIDForPageURL(int64_t, const String&);
    Vector<uint8_t> iconData(int64_t);
    std::optional<int64_t> addIcon(const String&, const Vector<uint8_t>&);
    void updateIconTimestamp(int64_t iconID, int64_t timestamp);
    void deleteIcon(int64_t);

    Ref<WorkQueue> m_workQueue;
    AllowDatabaseWrite m_allowDatabaseWrite { AllowDatabaseWrite::Yes };
    WebCore::SQLiteDatabase m_db;
    HashMap<String, String> m_pageURLToIconURLMap;
    Lock m_pageURLToIconURLMapLock;
    HashMap<String, std::pair<WebCore::PlatformImagePtr, MonotonicTime>> m_loadedIcons WTF_GUARDED_BY_LOCK(m_loadedIconsLock);
    Lock m_loadedIconsLock;

    std::unique_ptr<WebCore::SQLiteStatement> m_iconIDForIconURLStatement;
    std::unique_ptr<WebCore::SQLiteStatement> m_setIconIDForPageURLStatement;
    std::unique_ptr<WebCore::SQLiteStatement> m_iconDataStatement;
    std::unique_ptr<WebCore::SQLiteStatement> m_addIconStatement;
    std::unique_ptr<WebCore::SQLiteStatement> m_addIconDataStatement;
    std::unique_ptr<WebCore::SQLiteStatement> m_updateIconTimestampStatement;
    std::unique_ptr<WebCore::SQLiteStatement> m_deletePageURLsForIconStatement;
    std::unique_ptr<WebCore::SQLiteStatement> m_deleteIconDataStatement;
    std::unique_ptr<WebCore::SQLiteStatement> m_deleteIconStatement;
    std::unique_ptr<WebCore::SQLiteStatement> m_pruneIconsStatement;

    std::unique_ptr<RunLoop::Timer> m_pruneTimer;
    RunLoop::Timer m_clearLoadedIconsTimer;
};

} // namespace WebKit
