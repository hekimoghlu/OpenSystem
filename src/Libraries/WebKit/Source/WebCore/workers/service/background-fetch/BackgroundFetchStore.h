/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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

#include "ServiceWorkerRegistrationKey.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Expected.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class BackgroundFetchEngine;
class ResourceError;
class ResourceResponse;
class SWServerRegistration;
class SharedBuffer;

struct BackgroundFetchInformation;
struct BackgroundFetchOptions;
struct BackgroundFetchRecordInformation;
struct BackgroundFetchRequest;
struct ExceptionData;
struct RetrieveRecordsOptions;

class BackgroundFetchStore : public RefCountedAndCanMakeWeakPtr<BackgroundFetchStore> {
public:
    virtual ~BackgroundFetchStore() = default;

    virtual void initializeFetches(const ServiceWorkerRegistrationKey&, CompletionHandler<void()>&&) = 0;
    virtual void clearFetch(const ServiceWorkerRegistrationKey&, const String&, CompletionHandler<void()>&& = [] { }) = 0;
    virtual void clearAllFetches(const ServiceWorkerRegistrationKey&, CompletionHandler<void()>&& = [] { }) = 0;

    enum class StoreResult : uint8_t { OK, QuotaError, InternalError };
    virtual void storeFetch(const ServiceWorkerRegistrationKey&, const String&, uint64_t downloadTotal, uint64_t uploadTotal, std::optional<size_t> responseBodyIndexToClear, Vector<uint8_t>&&, CompletionHandler<void(StoreResult)>&&) = 0;
    virtual void storeFetchResponseBodyChunk(const ServiceWorkerRegistrationKey&, const String&, size_t, const SharedBuffer&, CompletionHandler<void(StoreResult)>&&) = 0;

    using RetrieveRecordResponseBodyCallback = Function<void(Expected<RefPtr<SharedBuffer>, ResourceError>&&)>;
    virtual void retrieveResponseBody(const ServiceWorkerRegistrationKey&, const String&, size_t, RetrieveRecordResponseBodyCallback&&) = 0;
};

} // namespace WebCore
