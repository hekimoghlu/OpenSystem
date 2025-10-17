/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 30, 2024.
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
#include "config.h"
#include "CacheStorageMemoryStore.h"

#include <WebCore/DOMCacheEngine.h>

namespace WebKit {

Ref<CacheStorageMemoryStore> CacheStorageMemoryStore::create()
{
    return adoptRef(*new CacheStorageMemoryStore);
}

static CacheStorageRecord copyCacheStorageRecord(const CacheStorageRecord& record)
{
    return { record.info, record.requestHeadersGuard, record.request, record.options, record.referrer, record.responseHeadersGuard, record.responseData.isolatedCopy(), record.responseBodySize, WebCore::DOMCacheEngine::copyResponseBody(record.responseBody) };
}

void CacheStorageMemoryStore::readAllRecordInfos(ReadAllRecordInfosCallback&& callback)
{
    callback(WTF::map(m_records.values(), [](const auto& record) {
        RELEASE_ASSERT(record);
        return record->info;
    }));
}

void CacheStorageMemoryStore::readRecords(const Vector<CacheStorageRecordInformation>& recordInfos, ReadRecordsCallback&& callback)
{
    auto result = WTF::map(recordInfos, [&](auto& recordInfo) -> std::optional<CacheStorageRecord> {
        auto iterator = m_records.find(recordInfo.identifier());
        if (iterator == m_records.end())
            return std::nullopt;
        return copyCacheStorageRecord(*iterator->value);
    });
    return callback(WTFMove(result));
}

void CacheStorageMemoryStore::deleteRecords(const Vector<CacheStorageRecordInformation>& recordInfos, WriteRecordsCallback&& callback)
{
    for (auto& recordInfo : recordInfos)
        m_records.remove(recordInfo.identifier());

    callback(true);
}

void CacheStorageMemoryStore::writeRecords(Vector<CacheStorageRecord>&& records, WriteRecordsCallback&& callback)
{
    for (auto&& record : records)
        m_records.set(record.info.identifier(), makeUnique<CacheStorageRecord>(WTFMove(record)));

    callback(true);
}

}

