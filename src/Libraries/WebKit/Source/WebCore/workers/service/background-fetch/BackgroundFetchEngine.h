/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 3, 2022.
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

#include "BackgroundFetch.h"
#include <span>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class BackgroundFetchStore;
class ResourceResponse;
class SWServer;

class BackgroundFetchEngine : public RefCountedAndCanMakeWeakPtr<BackgroundFetchEngine> {
    WTF_MAKE_TZONE_ALLOCATED(BackgroundFetchEngine);
public:
    static Ref<BackgroundFetchEngine> create(SWServer& swServer)
    {
        return adoptRef(*new BackgroundFetchEngine(swServer));
    }

    using ExceptionOrBackgroundFetchInformationCallback = CompletionHandler<void(Expected<std::optional<BackgroundFetchInformation>, ExceptionData>&&)>;
    void startBackgroundFetch(SWServerRegistration&, const String&, Vector<BackgroundFetchRequest>&&, BackgroundFetchOptions&&, ExceptionOrBackgroundFetchInformationCallback&&);
    void backgroundFetchInformation(SWServerRegistration&, const String&, ExceptionOrBackgroundFetchInformationCallback&&);
    using BackgroundFetchIdentifiersCallback = CompletionHandler<void(Vector<String>&&)>;
    void backgroundFetchIdentifiers(SWServerRegistration&, BackgroundFetchIdentifiersCallback&&);
    using AbortBackgroundFetchCallback = CompletionHandler<void(bool)>;
    void abortBackgroundFetch(SWServerRegistration&, const String&, AbortBackgroundFetchCallback&&);
    using MatchBackgroundFetchCallback = CompletionHandler<void(Vector<BackgroundFetchRecordInformation>&&)>;
    void matchBackgroundFetch(SWServerRegistration&, const String&, RetrieveRecordsOptions&&, MatchBackgroundFetchCallback&&);
    using RetrieveRecordResponseCallback = BackgroundFetch::RetrieveRecordResponseCallback;
    void retrieveRecordResponse(BackgroundFetchRecordIdentifier, RetrieveRecordResponseCallback&&);
    using RetrieveRecordResponseBodyCallback = BackgroundFetch::RetrieveRecordResponseBodyCallback;
    void retrieveRecordResponseBody(BackgroundFetchRecordIdentifier, RetrieveRecordResponseBodyCallback&&);

    void remove(SWServerRegistration&);

    WEBCORE_EXPORT WeakPtr<BackgroundFetch> backgroundFetch(const ServiceWorkerRegistrationKey&, const String&) const;
    WEBCORE_EXPORT void addFetchFromStore(std::span<const uint8_t>, CompletionHandler<void(const ServiceWorkerRegistrationKey&, const String&)>&&);

    WEBCORE_EXPORT void abortBackgroundFetch(const ServiceWorkerRegistrationKey&, const String&);
    WEBCORE_EXPORT void pauseBackgroundFetch(const ServiceWorkerRegistrationKey&, const String&);
    WEBCORE_EXPORT void resumeBackgroundFetch(const ServiceWorkerRegistrationKey&, const String&);
    WEBCORE_EXPORT void clickBackgroundFetch(const ServiceWorkerRegistrationKey&, const String&);

private:
    explicit BackgroundFetchEngine(SWServer&);

    void notifyBackgroundFetchUpdate(BackgroundFetch&);

    WeakPtr<SWServer> m_server;
    Ref<BackgroundFetchStore> m_store;

    using FetchesMap = HashMap<String, Ref<BackgroundFetch>>;
    HashMap<ServiceWorkerRegistrationKey, FetchesMap> m_fetches;

    HashMap<BackgroundFetchRecordIdentifier, Ref<BackgroundFetch::Record>> m_records;
};

} // namespace WebCore
