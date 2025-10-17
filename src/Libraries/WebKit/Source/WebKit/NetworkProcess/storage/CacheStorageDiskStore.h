/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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

#include "CacheStorageStore.h"
#include "NetworkCacheKey.h"
#include <wtf/FileSystem.h>
#include <wtf/WorkQueue.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class FormData;
class SharedBuffer;
namespace DOMCacheEngine {
using ResponseBody = std::variant<std::nullptr_t, Ref<FormData>, Ref<SharedBuffer>>;
}
}

namespace WebKit {

class CacheStorageDiskStore final : public CacheStorageStore {
public:
    static Ref<CacheStorageDiskStore> create(const String& cacheName, const String& path, Ref<WorkQueue>&&);
    static size_t computeRealBodySizeForStorage(const WebCore::DOMCacheEngine::ResponseBody&);

private:
    CacheStorageDiskStore(const String& cacheName, const String& path, Ref<WorkQueue>&&);

    // CacheStorageStore
    void readAllRecordInfos(ReadAllRecordInfosCallback&&) final;
    void readRecords(const Vector<CacheStorageRecordInformation>&, ReadRecordsCallback&&) final;
    void deleteRecords(const Vector<CacheStorageRecordInformation>&, WriteRecordsCallback&&) final;
    void writeRecords(Vector<CacheStorageRecord>&&, WriteRecordsCallback&&) final;

    String versionDirectoryPath() const;
    String saltFilePath() const;
    String recordsDirectoryPath() const;
    String recordFilePath(const NetworkCache::Key&) const;
    String recordBlobFilePath(const String&) const;
    String blobsDirectoryPath() const;
    String blobFilePath(const String&) const;
    std::optional<CacheStorageRecord> readRecordFromFileData(std::span<const uint8_t>, FileSystem::MappedFileData&&);
    void readAllRecordInfosInternal(ReadAllRecordInfosCallback&&);
    void readRecordsInternal(const Vector<CacheStorageRecordInformation>&, ReadRecordsCallback&&);

    String m_cacheName;
    String m_path;
    FileSystem::Salt m_salt;
    Ref<WorkQueue> m_callbackQueue;
    Ref<WorkQueue> m_ioQueue;
};

}

