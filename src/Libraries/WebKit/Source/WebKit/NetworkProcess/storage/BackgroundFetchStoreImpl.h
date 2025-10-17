/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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

#include <WebCore/BackgroundFetchStore.h>
#include <WebCore/ClientOrigin.h>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>

namespace WTF {
class WorkQueue;
}

namespace WebCore {
class SWServer;
}

namespace WebKit {

class NetworkStorageManager;
struct BackgroundFetchState;

class BackgroundFetchStoreImpl :  public WebCore::BackgroundFetchStore {
public:
    static Ref<BackgroundFetchStoreImpl> create(WeakPtr<NetworkStorageManager>&& manager, WeakPtr<WebCore::SWServer>&& server) { return adoptRef(*new BackgroundFetchStoreImpl(WTFMove(manager), WTFMove(server))); }
    ~BackgroundFetchStoreImpl();

    void getAllBackgroundFetchIdentifiers(CompletionHandler<void(Vector<String>&&)>&&);
    void getBackgroundFetchState(const String&, CompletionHandler<void(std::optional<BackgroundFetchState>&&)>&&);
    void abortBackgroundFetch(const String&, CompletionHandler<void()>&&);
    void pauseBackgroundFetch(const String&, CompletionHandler<void()>&&);
    void resumeBackgroundFetch(const String&, CompletionHandler<void()>&&);
    void clickBackgroundFetch(const String&, CompletionHandler<void()>&&);

private:
    BackgroundFetchStoreImpl(WeakPtr<NetworkStorageManager>&&, WeakPtr<WebCore::SWServer>&&);

    void initializeFetches(const WebCore::ServiceWorkerRegistrationKey&, CompletionHandler<void()>&&) final;
    void clearFetch(const WebCore::ServiceWorkerRegistrationKey&, const String&, CompletionHandler<void()>&&) final;
    void clearAllFetches(const WebCore::ServiceWorkerRegistrationKey&, CompletionHandler<void()>&&) final;
    void storeFetch(const WebCore::ServiceWorkerRegistrationKey&, const String&, uint64_t downloadTotal, uint64_t uploadTotal, std::optional<size_t> responseBodyIndexToClear, Vector<uint8_t>&&, CompletionHandler<void(StoreResult)>&&) final;
    void storeFetchResponseBodyChunk(const WebCore::ServiceWorkerRegistrationKey&, const String&, size_t, const WebCore::SharedBuffer&, CompletionHandler<void(StoreResult)>&&) final;
    void retrieveResponseBody(const WebCore::ServiceWorkerRegistrationKey&, const String&, size_t, RetrieveRecordResponseBodyCallback&&) final;

    void initializeFetchesInternal(const WebCore::ClientOrigin&, CompletionHandler<void(Vector<std::pair<RefPtr<WebCore::SharedBuffer>, String>>&&)>&&);
    void clearFetchInternal(const WebCore::ClientOrigin&, const String&, CompletionHandler<void()>&&);
    void clearAllFetchesInternal(const WebCore::ClientOrigin&, const Vector<String>&, CompletionHandler<void()>&&);
    void storeFetchInternal(const WebCore::ClientOrigin&, const String&, uint64_t, uint64_t, std::optional<size_t>, Vector<uint8_t>&&, CompletionHandler<void(StoreResult)>&&);
    void storeFetchResponseBodyChunkInternal(const WebCore::ClientOrigin&, const String&, size_t index, const WebCore::SharedBuffer&, CompletionHandler<void(StoreResult)>&&);

    String getFilename(const WebCore::ServiceWorkerRegistrationKey&, const String&);
    void registerFetch(const WebCore::ClientOrigin&, const WebCore::ServiceWorkerRegistrationKey&, const String& backgroundFetchIdentifier, String&& fetchStorageIdentifier);
    void loadAllFetches(CompletionHandler<void()>&&);
    void fetchInformationFromFilename(const String&, CompletionHandler<void(const WebCore::ServiceWorkerRegistrationKey&, const String&)>&&);
    void initializeFetches(const WebCore::ClientOrigin&, CompletionHandler<void()>&&);

    WeakPtr<NetworkStorageManager> m_manager;

    using FetchIdentifier = std::pair<String, String>; // < service worker registration scope, background fetch identifier >
    struct PerClientOriginFetches {
        HashMap<FetchIdentifier, String> fetchToFilenames;
        Vector<CompletionHandler<void()>> initializationCallbacks;
    };
    HashMap<WebCore::ClientOrigin, PerClientOriginFetches> m_perClientOriginFetches;

    struct FetchInformation {
        WebCore::ClientOrigin origin;
        FetchIdentifier identifier;
    };
    HashMap<String, FetchInformation> m_filenameToFetch;
    WeakPtr<WebCore::SWServer> m_server;
};

} // namespace WebKit
