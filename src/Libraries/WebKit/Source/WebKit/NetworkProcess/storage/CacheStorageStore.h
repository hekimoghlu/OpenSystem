/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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

#include <optional>
#include <wtf/Forward.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebKit {

class CacheStorageRecordInformation;
struct CacheStorageRecord;

class CacheStorageStore : public ThreadSafeRefCounted<CacheStorageStore> {
public:
    virtual ~CacheStorageStore() = default;
    using ReadAllRecordInfosCallback = CompletionHandler<void(Vector<CacheStorageRecordInformation>&&)>;
    using ReadRecordsCallback = CompletionHandler<void(Vector<std::optional<CacheStorageRecord>>&&)>;
    using WriteRecordsCallback = CompletionHandler<void(bool)>;
    virtual void readAllRecordInfos(ReadAllRecordInfosCallback&&) = 0;
    virtual void readRecords(const Vector<CacheStorageRecordInformation>&, ReadRecordsCallback&&) = 0;
    virtual void deleteRecords(const Vector<CacheStorageRecordInformation>&, WriteRecordsCallback&&) = 0;
    virtual void writeRecords(Vector<CacheStorageRecord>&&, WriteRecordsCallback&&) = 0;

protected:
    CacheStorageStore() = default;
};

} // namespace WebKit

