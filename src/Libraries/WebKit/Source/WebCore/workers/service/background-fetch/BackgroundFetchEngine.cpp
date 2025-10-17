/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#include "BackgroundFetchEngine.h"

#include "BackgroundFetchInformation.h"
#include "BackgroundFetchRecordInformation.h"
#include "ExceptionData.h"
#include "Logging.h"
#include "RetrieveRecordsOptions.h"
#include "SWServerRegistration.h"
#include "SWServerToContextConnection.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BackgroundFetchEngine);

BackgroundFetchEngine::BackgroundFetchEngine(SWServer& server)
    : m_server(server)
    , m_store(server.createBackgroundFetchStore())
{
}

// https://wicg.github.io/background-fetch/#dom-backgroundfetchmanager-fetch starting from step 8.3
void BackgroundFetchEngine::startBackgroundFetch(SWServerRegistration& registration, const String& backgroundFetchIdentifier, Vector<BackgroundFetchRequest>&& requests, BackgroundFetchOptions&& options, ExceptionOrBackgroundFetchInformationCallback&& callback)
{
    auto iterator = m_fetches.find(registration.key());
    if (iterator == m_fetches.end()) {
        m_store->initializeFetches(registration.key(), [weakThis = WeakPtr { *this }, registration = WeakPtr { registration }, backgroundFetchIdentifier, requests = WTFMove(requests), options = WTFMove(options), callback = WTFMove(callback)]() mutable {
            if (!weakThis || !registration) {
                callback(makeUnexpected(ExceptionData { ExceptionCode::InvalidStateError, "BackgroundFetchEngine is gone"_s }));
                return;
            }
            weakThis->m_fetches.ensure(registration->key(), [] {
                return FetchesMap();
            });
            weakThis->startBackgroundFetch(*registration, backgroundFetchIdentifier, WTFMove(requests), WTFMove(options), WTFMove(callback));
        });
        return;
    }

    auto result = iterator->value.ensure(backgroundFetchIdentifier, [&]() {
        return BackgroundFetch::create(registration, backgroundFetchIdentifier, WTFMove(requests), WTFMove(options), Ref { m_store }, [weakThis = WeakPtr { *this }](auto& fetch) {
            if (weakThis)
                weakThis->notifyBackgroundFetchUpdate(fetch);
        });
    });
    if (!result.isNewEntry) {
        callback(makeUnexpected(ExceptionData { ExceptionCode::TypeError, "A background fetch registration already exists"_s }));
        return;
    }

    auto fetch = result.iterator->value;
    fetch->doStore([server = m_server, fetch = WeakPtr { fetch }, callback = WTFMove(callback)](auto result) mutable {
        if (!fetch || !server) {
            callback(makeUnexpected(ExceptionData { ExceptionCode::TypeError, "Background fetch is gone"_s }));
            return;
        }
        switch (result) {
        case BackgroundFetchStore::StoreResult::QuotaError:
            callback(makeUnexpected(ExceptionData { ExceptionCode::QuotaExceededError, "Background fetch requested space is above quota"_s }));
            break;
        case BackgroundFetchStore::StoreResult::InternalError:
            callback(makeUnexpected(ExceptionData { ExceptionCode::TypeError, "Background fetch store operation failed"_s }));
            break;
        case BackgroundFetchStore::StoreResult::OK:
            if (!fetch->pausedFlagIsSet()) {
                fetch->perform([server = WTFMove(server)](auto& client, auto& request, auto responseDataSize, auto& origin) mutable {
                    return server ? RefPtr { server->createBackgroundFetchRecordLoader(client, request, responseDataSize, origin) } : nullptr;
                });
            }
            callback(std::optional { fetch->information() });
            break;
        };
    });
}

// https://wicg.github.io/background-fetch/#update-background-fetch-instance-algorithm
void BackgroundFetchEngine::notifyBackgroundFetchUpdate(BackgroundFetch& fetch)
{
    auto information = fetch.information();
    auto* registration = m_server->getRegistration(information.registrationIdentifier);
    if (!registration)
        return;

    // Progress event.
    registration->forEachConnection([&](auto& connection) {
        connection.updateBackgroundFetchRegistration(information);
    });

    if (information.result == BackgroundFetchResult::EmptyString || !information.recordsAvailable)
        return;

    // FIXME: We should delay events if the service worker (or related page) is not running.
    m_server->fireBackgroundFetchEvent(*registration, WTFMove(information), [weakFetch = WeakPtr { fetch }]() {
        if (weakFetch)
            weakFetch->unsetRecordsAvailableFlag();
    });
}

void BackgroundFetchEngine::backgroundFetchInformation(SWServerRegistration& registration, const String& backgroundFetchIdentifier, ExceptionOrBackgroundFetchInformationCallback&& callback)
{
    auto iterator = m_fetches.find(registration.key());
    if (iterator == m_fetches.end()) {
        m_store->initializeFetches(registration.key(), [weakThis = WeakPtr { *this }, registration = WeakPtr { registration }, backgroundFetchIdentifier, callback = WTFMove(callback)]() mutable {
            if (!weakThis || !registration) {
                callback(makeUnexpected(ExceptionData { ExceptionCode::InvalidStateError, "BackgroundFetchEngine is gone"_s }));
                return;
            }
            weakThis->m_fetches.ensure(registration->key(), [] {
                return FetchesMap();
            });
            weakThis->backgroundFetchInformation(*registration, backgroundFetchIdentifier, WTFMove(callback));
        });
        return;
    }

    auto& map = iterator->value;
    auto fetchIterator = map.find(backgroundFetchIdentifier);
    if (fetchIterator == map.end()) {
        callback(std::optional<BackgroundFetchInformation> { });
        return;
    }
    callback(std::optional { fetchIterator->value->information() });
}

// https://wicg.github.io/background-fetch/#dom-backgroundfetchmanager-getids
void BackgroundFetchEngine::backgroundFetchIdentifiers(SWServerRegistration& registration, BackgroundFetchIdentifiersCallback&& callback)
{
    auto iterator = m_fetches.find(registration.key());
    if (iterator == m_fetches.end()) {
        m_store->initializeFetches(registration.key(), [weakThis = WeakPtr { *this }, registration = WeakPtr { registration }, callback = WTFMove(callback)]() mutable {
            if (!weakThis || !registration) {
                callback({ });
                return;
            }
            weakThis->m_fetches.ensure(registration->key(), [] {
                return FetchesMap();
            });
            weakThis->backgroundFetchIdentifiers(*registration, WTFMove(callback));
        });
        return;
    }

    Vector<String> identifiers = WTF::compactMap(iterator->value, [](auto& keyValue) -> std::optional<String> {
        if (keyValue.value->isActive())
            return keyValue.key;
        return std::nullopt;
    });
    callback(WTFMove(identifiers));
}

// https://wicg.github.io/background-fetch/#background-fetch-registration-abort starting from step 3
void BackgroundFetchEngine::abortBackgroundFetch(SWServerRegistration& registration, const String& backgroundFetchIdentifier, AbortBackgroundFetchCallback&& callback)
{
    auto iterator = m_fetches.find(registration.key());
    if (iterator == m_fetches.end()) {
        m_store->initializeFetches(registration.key(), [weakThis = WeakPtr { *this }, registration = WeakPtr { registration }, backgroundFetchIdentifier, callback = WTFMove(callback)]() mutable {
            if (!weakThis || !registration) {
                callback(false);
                return;
            }
            weakThis->m_fetches.ensure(registration->key(), [] {
                return FetchesMap();
            });
            weakThis->abortBackgroundFetch(*registration, backgroundFetchIdentifier, WTFMove(callback));
        });
        return;
    }

    auto& map = iterator->value;
    auto fetchIterator = map.find(backgroundFetchIdentifier);
    if (fetchIterator == map.end()) {
        callback(false);
        return;
    }
    callback(fetchIterator->value->abort());
}

// https://wicg.github.io/background-fetch/#dom-backgroundfetchregistration-matchall starting from step 3
void BackgroundFetchEngine::matchBackgroundFetch(SWServerRegistration& registration, const String& backgroundFetchIdentifier, RetrieveRecordsOptions&& options, MatchBackgroundFetchCallback&& callback)
{
    auto iterator = m_fetches.find(registration.key());
    if (iterator == m_fetches.end()) {
        m_store->initializeFetches(registration.key(), [weakThis = WeakPtr { *this }, registration = WeakPtr { registration }, backgroundFetchIdentifier, options = WTFMove(options), callback = WTFMove(callback)]() mutable {
            if (!weakThis || !registration) {
                callback({ });
                return;
            }
            weakThis->m_fetches.ensure(registration->key(), [] {
                return FetchesMap();
            });
            weakThis->matchBackgroundFetch(*registration, backgroundFetchIdentifier, WTFMove(options), WTFMove(callback));
        });
        return;
    }

    auto& map = iterator->value;
    auto fetchIterator = map.find(backgroundFetchIdentifier);
    if (fetchIterator == map.end()) {
        callback({ });
        return;
    }
    fetchIterator->value->match(options, [weakThis = WeakPtr { *this }, callback = WTFMove(callback)](auto&& records) mutable {
        if (!weakThis) {
            callback({ });
            return;
        }
        auto recordsInformation = WTF::map(WTFMove(records), [&](auto&& record) {
            // FIXME: We need a way to remove the record from m_records.
            auto information = record->information();
            weakThis->m_records.add(information.identifier, WTFMove(record));
            return information;
        });
        callback(WTFMove(recordsInformation));
    });
}

void BackgroundFetchEngine::remove(SWServerRegistration& registration)
{
    // FIXME: We skip the initialization step, which might invalidate some results, maybe we should have a specific handling here.
    auto fetches = m_fetches.take(registration.key());
    for (auto& fetch : fetches.values())
        fetch->abort();
    m_store->clearAllFetches(registration.key());
}

void BackgroundFetchEngine::retrieveRecordResponse(BackgroundFetchRecordIdentifier recordIdentifier, RetrieveRecordResponseCallback&& callback)
{
    auto record = m_records.get(recordIdentifier);
    if (!record) {
        callback(makeUnexpected(ExceptionData { ExceptionCode::InvalidStateError, "Record not found"_s }));
        return;
    }
    record->retrieveResponse(m_store.get(), WTFMove(callback));
}

void BackgroundFetchEngine::retrieveRecordResponseBody(BackgroundFetchRecordIdentifier recordIdentifier, RetrieveRecordResponseBodyCallback&& callback)
{
    auto record = m_records.get(recordIdentifier);
    if (!record) {
        callback(makeUnexpected(ResourceError { errorDomainWebKitInternal, 0, { }, "Record not found"_s }));
        return;
    }
    record->retrieveRecordResponseBody(m_store.get(), WTFMove(callback));
}

void BackgroundFetchEngine::addFetchFromStore(std::span<const uint8_t> data, CompletionHandler<void(const ServiceWorkerRegistrationKey&, const String&)>&& callback)
{
    auto fetch = BackgroundFetch::createFromStore(data, *m_server, m_store.get(), [weakThis = WeakPtr { *this }](auto& fetch) {
        if (weakThis)
            weakThis->notifyBackgroundFetchUpdate(fetch);
    });
    if (!fetch) {
        RELEASE_LOG_ERROR(ServiceWorker, "BackgroundFetchEngine failed adding fetch entry as registration is missing");
        callback({ }, { });
        return;
    }

    callback(fetch->registrationKey(), fetch->identifier());

    auto& fetchMap = m_fetches.ensure(fetch->registrationKey(), [] {
        return FetchesMap();
    }).iterator->value;

    auto backgroundFetchIdentifier = fetch->identifier();
    ASSERT(!fetchMap.contains(backgroundFetchIdentifier));
    fetchMap.add(WTFMove(backgroundFetchIdentifier), fetch.releaseNonNull());
}

void BackgroundFetchEngine::abortBackgroundFetch(const ServiceWorkerRegistrationKey& key, const String& identifier)
{
    if (auto *registration = m_server ? m_server->getRegistration(key) : nullptr)
        abortBackgroundFetch(*registration, identifier, [](auto) { });
}

void BackgroundFetchEngine::pauseBackgroundFetch(const ServiceWorkerRegistrationKey& key, const String& identifier)
{
    auto* registration = m_server ? m_server->getRegistration(key) : nullptr;
    if (!registration)
        return;

    auto iterator = m_fetches.find(key);
    if (iterator == m_fetches.end())
        return;

    auto& map = iterator->value;
    auto fetchIterator = map.find(identifier);
    if (fetchIterator == map.end())
        return;

    fetchIterator->value->pause();
}

void BackgroundFetchEngine::resumeBackgroundFetch(const ServiceWorkerRegistrationKey& key, const String& identifier)
{
    auto* registration = m_server ? m_server->getRegistration(key) : nullptr;
    if (!registration)
        return;

    auto iterator = m_fetches.find(key);
    if (iterator == m_fetches.end())
        return;

    auto& map = iterator->value;
    auto fetchIterator = map.find(identifier);
    if (fetchIterator == map.end())
        return;

    fetchIterator->value->resume([server = m_server](auto& client, auto& request, auto responseDataSize, auto& origin) mutable {
        return server ? RefPtr { server->createBackgroundFetchRecordLoader(client, request, responseDataSize, origin) } : nullptr;
    });
}

void BackgroundFetchEngine::clickBackgroundFetch(const ServiceWorkerRegistrationKey& key, const String& backgroundFetchIdentifier)
{
    auto* registration = m_server ? m_server->getRegistration(key) : nullptr;
    if (!registration)
        return;

    auto iterator = m_fetches.find(key);
    if (iterator == m_fetches.end())
        return;

    auto& map = iterator->value;
    auto fetchIterator = map.find(backgroundFetchIdentifier);
    if (fetchIterator == map.end())
        return;

    m_server->fireBackgroundFetchClickEvent(*registration, fetchIterator->value->information());
}

WeakPtr<BackgroundFetch> BackgroundFetchEngine::backgroundFetch(const ServiceWorkerRegistrationKey& key, const String& identifier) const
{
    auto iterator = m_fetches.find(key);
    if (iterator == m_fetches.end())
        return { };

    return iterator->value.get(identifier);
}

} // namespace WebCore
