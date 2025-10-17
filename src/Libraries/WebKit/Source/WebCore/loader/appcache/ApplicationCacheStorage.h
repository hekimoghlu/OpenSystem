/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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

#include "SecurityOriginHash.h"
#include "SQLiteDatabase.h"
#include <wtf/HashCountedSet.h>
#include <wtf/HashSet.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ApplicationCache;
class ApplicationCacheGroup;
class ApplicationCacheHost;
class ApplicationCacheResource;
class SecurityOrigin;
class FragmentedSharedBuffer;
template<typename> class StorageIDJournal;

class ApplicationCacheStorage : public RefCounted<ApplicationCacheStorage> {
public:
    enum FailureReason {
        OriginQuotaReached,
        TotalQuotaReached,
        DiskOrOperationFailure
    };

    static Ref<ApplicationCacheStorage> create(const String& cacheDirectory, const String& flatFileSubdirectoryName)
    {
        return adoptRef(*new ApplicationCacheStorage(cacheDirectory, flatFileSubdirectoryName));
    }


    const String& cacheDirectory() const { return m_cacheDirectory; }
    WEBCORE_EXPORT void setMaximumSize(int64_t size);
    WEBCORE_EXPORT int64_t maximumSize() const;
    bool isMaximumSizeReached() const;
    int64_t spaceNeeded(int64_t cacheToSave);

    int64_t defaultOriginQuota() const { return m_defaultOriginQuota; }
    WEBCORE_EXPORT void setDefaultOriginQuota(int64_t quota);
    WEBCORE_EXPORT bool calculateUsageForOrigin(const SecurityOriginData&, int64_t& usage);
    WEBCORE_EXPORT bool calculateQuotaForOrigin(const SecurityOrigin&, int64_t& quota);
    bool calculateRemainingSizeForOriginExcludingCache(const SecurityOrigin&, ApplicationCache*, int64_t& remainingSize);
    WEBCORE_EXPORT bool storeUpdatedQuotaForOrigin(const SecurityOrigin*, int64_t quota);
    bool checkOriginQuota(ApplicationCacheGroup*, ApplicationCache* oldCache, ApplicationCache* newCache, int64_t& totalSpaceNeeded);

    ApplicationCacheGroup* cacheGroupForURL(const URL&); // Cache to load a main resource from.
    ApplicationCacheGroup* fallbackCacheGroupForURL(const URL&); // Cache that has a fallback entry to load a main resource from if normal loading fails.

    ApplicationCacheGroup* findOrCreateCacheGroup(const URL& manifestURL);
    void cacheGroupDestroyed(ApplicationCacheGroup&);
    void cacheGroupMadeObsolete(ApplicationCacheGroup&);

    bool storeNewestCache(ApplicationCacheGroup&, ApplicationCache* oldCache, FailureReason&);
    bool storeNewestCache(ApplicationCacheGroup&); // Updates the cache group, but doesn't remove old cache.
    bool store(ApplicationCacheResource*, ApplicationCache*);
    bool storeUpdatedType(ApplicationCacheResource*, ApplicationCache*);

    // Removes the group if the cache to be removed is the newest one (so, storeNewestCache() needs to be called beforehand when updating).
    void remove(ApplicationCache*);

    WEBCORE_EXPORT void empty();

    WEBCORE_EXPORT UncheckedKeyHashSet<SecurityOriginData> originsWithCache();
    WEBCORE_EXPORT void deleteAllEntries();

    // FIXME: This should be consolidated with deleteAllEntries().
    WEBCORE_EXPORT void deleteAllCaches();

    // FIXME: This should be consolidated with deleteCacheGroup().
    WEBCORE_EXPORT void deleteCacheForOrigin(const SecurityOriginData&);

    // FIXME: This should be consolidated with calculateUsageForOrigin().
    WEBCORE_EXPORT int64_t diskUsageForOrigin(const SecurityOriginData&);

    static int64_t unknownQuota() { return -1; }
    static int64_t noQuota() { return std::numeric_limits<int64_t>::max(); }

private:
    WEBCORE_EXPORT ApplicationCacheStorage(const String& cacheDirectory, const String& flatFileSubdirectoryName);

    RefPtr<ApplicationCache> loadCache(unsigned storageID);
    ApplicationCacheGroup* loadCacheGroup(const URL& manifestURL);
    std::optional<Vector<URL>> manifestURLs();
    ApplicationCacheGroup* findInMemoryCacheGroup(const URL& manifestURL) const;
    bool deleteCacheGroup(const String& manifestURL);
    void vacuumDatabaseFile();
    
    using ResourceStorageIDJournal = StorageIDJournal<ApplicationCacheResource>;
    using GroupStorageIDJournal = StorageIDJournal<ApplicationCacheGroup>;

    bool store(ApplicationCacheGroup*, GroupStorageIDJournal*);
    bool store(ApplicationCache*, ResourceStorageIDJournal*);
    bool store(ApplicationCacheResource*, unsigned cacheStorageID);
    bool deleteCacheGroupRecord(const String& manifestURL);

    bool ensureOriginRecord(const SecurityOrigin*);
    static bool shouldStoreResourceAsFlatFile(ApplicationCacheResource*);
    void deleteTables();
    bool writeDataToUniqueFileInDirectory(FragmentedSharedBuffer&, const String& directory, String& outFilename, StringView fileExtension);

    void loadManifestHostHashes();
    
    void verifySchemaVersion();
    
    void openDatabase(bool createIfDoesNotExist);
    
    bool executeStatement(SQLiteStatement&);
    bool executeSQLCommand(ASCIILiteral);

    void checkForMaxSizeReached();
    void checkForDeletedResources();
    long long flatFileAreaSize();

    const String m_cacheDirectory;
    const String m_flatFileSubdirectoryName;
    String m_cacheFile;

    int64_t m_maximumSize { noQuota() };
    bool m_isMaximumSizeReached { false };

    int64_t m_defaultOriginQuota { noQuota() };

    SQLiteDatabase m_database;

    // In order to quickly determine if a given resource exists in an application cache,
    // we keep a hash set of the hosts of the manifest URLs of all non-obsolete cache groups.
    HashCountedSet<unsigned, AlreadyHashed> m_cacheHostSet;
    
    HashMap<String, ApplicationCacheGroup*> m_cachesInMemory; // Excludes obsolete cache groups.

    friend class NeverDestroyed<ApplicationCacheStorage>;
};

} // namespace WebCore
