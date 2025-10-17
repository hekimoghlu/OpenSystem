/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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

#include "BackgroundFetchFailureReason.h"
#include "BackgroundFetchOptions.h"
#include "BackgroundFetchRecordIdentifier.h"
#include "BackgroundFetchRecordLoader.h"
#include "BackgroundFetchRequest.h"
#include "BackgroundFetchResult.h"
#include "BackgroundFetchStore.h"
#include "ClientOrigin.h"
#include "ResourceResponse.h"
#include "ServiceWorkerRegistrationKey.h"
#include "ServiceWorkerTypes.h"
#include <wtf/Identified.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class BackgroundFetchRecordLoader;
class SWServer;
class SharedBuffer;

struct BackgroundFetchRequest;
struct CacheQueryOptions;

class BackgroundFetch : public RefCountedAndCanMakeWeakPtr<BackgroundFetch> {
    WTF_MAKE_TZONE_ALLOCATED(BackgroundFetch);
public:
    using NotificationCallback = Function<void(BackgroundFetch&)>;

    static Ref<BackgroundFetch> create(SWServerRegistration& sWServerRegistration, const String& identifier, Vector<BackgroundFetchRequest>&& requests, BackgroundFetchOptions&& options, Ref<BackgroundFetchStore>&& store, NotificationCallback&& notificationCallback)
    {
        return adoptRef(*new BackgroundFetch(sWServerRegistration, identifier, WTFMove(requests), WTFMove(options), WTFMove(store), WTFMove(notificationCallback)));
    }

    static Ref<BackgroundFetch> create(SWServerRegistration& swServerRegistration, String&& identifier, BackgroundFetchOptions&& options, Ref<BackgroundFetchStore>&& store, NotificationCallback&& notificationCallback, bool pausedFlag)
    {
        return adoptRef(*new BackgroundFetch(swServerRegistration, WTFMove(identifier), WTFMove(options), WTFMove(store), WTFMove(notificationCallback), pausedFlag));
    }

    ~BackgroundFetch();

    static RefPtr<BackgroundFetch> createFromStore(std::span<const uint8_t>, SWServer&, Ref<BackgroundFetchStore>&&, NotificationCallback&&);

    String identifier() const { return m_identifier; }
    WEBCORE_EXPORT BackgroundFetchInformation information() const;
    const ServiceWorkerRegistrationKey& registrationKey() const { return m_registrationKey; }
    const BackgroundFetchOptions& options() const { return m_options; }

    using RetrieveRecordResponseCallback = CompletionHandler<void(Expected<ResourceResponse, ExceptionData>&&)>;
    using RetrieveRecordResponseBodyCallback = Function<void(Expected<RefPtr<SharedBuffer>, ResourceError>&&)>;
    using CreateLoaderCallback = Function<RefPtr<BackgroundFetchRecordLoader>(BackgroundFetchRecordLoaderClient&, const BackgroundFetchRequest&, size_t responseDataSize, const ClientOrigin&)>;

    bool pausedFlagIsSet() const { return m_pausedFlag; }
    void pause();
    void resume(const CreateLoaderCallback&);

    class Record final : public BackgroundFetchRecordLoaderClient, public RefCounted<Record>, private Identified<BackgroundFetchRecordIdentifier> {
        WTF_MAKE_TZONE_ALLOCATED(Record);
    public:
        void ref() const final { RefCounted::ref(); }
        void deref() const final { RefCounted::deref(); }

        static Ref<Record> create(BackgroundFetch& fetch, BackgroundFetchRequest&& request, size_t size) { return adoptRef(*new Record(fetch, WTFMove(request), size)); }
        ~Record();

        void complete(const CreateLoaderCallback&);
        void pause();
        void abort();

        void setAsCompleted() { m_isCompleted = true; }
        bool isCompleted() const { return m_isCompleted; }

        const BackgroundFetchRequest& request() const { return m_request; }
        const ResourceResponse& response() const { return m_response; }

        uint64_t responseDataSize() const { return m_responseDataSize; }
        void clearResponseDataSize() { m_responseDataSize = 0; }
        bool isMatching(const ResourceRequest&, const CacheQueryOptions&) const;
        BackgroundFetchRecordInformation information() const;

        void retrieveResponse(BackgroundFetchStore&, RetrieveRecordResponseCallback&&);
        void retrieveRecordResponseBody(BackgroundFetchStore&, RetrieveRecordResponseBodyCallback&&);

    private:
        Record(BackgroundFetch&, BackgroundFetchRequest&&, size_t);

        void didSendData(uint64_t) final;
        void didReceiveResponse(ResourceResponse&&) final;
        void didReceiveResponseBodyChunk(const SharedBuffer&) final;
        void didFinish(const ResourceError&) final;

        WeakPtr<BackgroundFetch> m_fetch;
        String m_fetchIdentifier;
        ServiceWorkerRegistrationKey m_registrationKey;
        BackgroundFetchRequest m_request;
        size_t m_index { 0 };
        ResourceResponse m_response;
        RefPtr<BackgroundFetchRecordLoader> m_loader;
        uint64_t m_responseDataSize { 0 };
        bool m_isCompleted { false };
        bool m_isAborted { false };
        Vector<RetrieveRecordResponseCallback> m_responseCallbacks;
        Vector<RetrieveRecordResponseBodyCallback> m_responseBodyCallbacks;
    };

    using MatchBackgroundFetchCallback = CompletionHandler<void(Vector<Ref<Record>>&&)>;
    void match(const RetrieveRecordsOptions&, MatchBackgroundFetchCallback&&);

    bool abort();

    void perform(const CreateLoaderCallback&);

    bool isActive() const { return m_isActive; }
    const ClientOrigin& origin() const { return m_origin; }
    uint64_t downloadTotal() const { return  m_options.downloadTotal; }
    uint64_t uploadTotal() const { return m_uploadTotal; }

    void doStore(CompletionHandler<void(BackgroundFetchStore::StoreResult)>&&, std::optional<size_t> responseBodyIndexToClear = { });
    void unsetRecordsAvailableFlag();

private:
    BackgroundFetch(SWServerRegistration&, const String&, Vector<BackgroundFetchRequest>&&, BackgroundFetchOptions&&, Ref<BackgroundFetchStore>&&, NotificationCallback&&);
    BackgroundFetch(SWServerRegistration&, String&&, BackgroundFetchOptions&&, Ref<BackgroundFetchStore>&&, NotificationCallback&&, bool pausedFlag);

    void didSendData(uint64_t);
    void storeResponse(size_t, bool shouldClearResponseBody, ResourceResponse&&);
    void storeResponseBodyChunk(size_t, const SharedBuffer&);
    void didFinishRecord(const ResourceError&);

    void recordIsCompleted();
    void handleStoreResult(BackgroundFetchStore::StoreResult);
    void updateBackgroundFetchStatus(BackgroundFetchResult, BackgroundFetchFailureReason);

    void setRecords(Vector<Ref<Record>>&&);

    String m_identifier;
    Vector<Ref<Record>> m_records;
    BackgroundFetchOptions m_options;
    ServiceWorkerRegistrationKey m_registrationKey;
    ServiceWorkerRegistrationIdentifier m_registrationIdentifier;

    BackgroundFetchResult m_result { BackgroundFetchResult::EmptyString };
    BackgroundFetchFailureReason m_failureReason { BackgroundFetchFailureReason::EmptyString };

    bool m_recordsAvailableFlag { true };
    bool m_abortFlag { false };
    bool m_pausedFlag { false };
    bool m_isActive { true };

    uint64_t m_uploadTotal { 0 };
    uint64_t m_currentDownloadSize { 0 };
    uint64_t m_currentUploadSize { 0 };

    Ref<BackgroundFetchStore> m_store;
    NotificationCallback m_notificationCallback;
    ClientOrigin m_origin;
};

} // namespace WebCore
