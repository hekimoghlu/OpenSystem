/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 29, 2025.
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
#include "PrivateClickMeasurementPersistentStore.h"

#include "PrivateClickMeasurementDatabase.h"
#include "PrivateClickMeasurementDebugInfo.h"
#include "PrivateClickMeasurementManager.h"
#include <WebCore/PrivateClickMeasurement.h>
#include <wtf/RunLoop.h>
#include <wtf/SuspendableWorkQueue.h>

namespace WebKit::PCM {

static Ref<SuspendableWorkQueue> sharedWorkQueue()
{
    static NeverDestroyed<Ref<SuspendableWorkQueue>> queue(SuspendableWorkQueue::create("PrivateClickMeasurement Process Data Queue"_s,  WorkQueue::QOS::Utility));
    return queue.get().copyRef();
}

void PersistentStore::prepareForProcessToSuspend(CompletionHandler<void()>&& completionHandler)
{
    ASSERT(RunLoop::isMain());
    sharedWorkQueue()->suspend(Database::interruptAllDatabases, WTFMove(completionHandler));
}

void PersistentStore::processDidResume()
{
    ASSERT(RunLoop::isMain());
    sharedWorkQueue()->resume();
}

PersistentStore::PersistentStore(const String& databaseDirectory)
    : m_queue(sharedWorkQueue())
{
    if (!databaseDirectory.isEmpty()) {
        postTask([this, protectedThis = Ref { *this }, databaseDirectory = databaseDirectory.isolatedCopy()] () mutable {
            m_database = Database::create(WTFMove(databaseDirectory));
        });
    }
}

PersistentStore::~PersistentStore() = default;

void PersistentStore::postTask(Function<void()>&& task) const
{
    ASSERT(RunLoop::isMain());
    m_queue->dispatch(WTFMove(task));
}

void PersistentStore::postTaskReply(WTF::Function<void()>&& reply) const
{
    ASSERT(!RunLoop::isMain());
    RunLoop::main().dispatch(WTFMove(reply));
}

void PersistentStore::insertPrivateClickMeasurement(WebCore::PrivateClickMeasurement&& attribution, PrivateClickMeasurementAttributionType attributionType, CompletionHandler<void()>&& completionHandler)
{
    postTask([this, protectedThis = Ref { *this }, attribution = WTFMove(attribution), attributionType, completionHandler = WTFMove(completionHandler)] () mutable {
        if (RefPtr database = m_database)
            database->insertPrivateClickMeasurement(WTFMove(attribution), attributionType);
        postTaskReply(WTFMove(completionHandler));
    });
}

void PersistentStore::markAllUnattributedPrivateClickMeasurementAsExpiredForTesting()
{
    postTask([this, protectedThis = Ref { *this }] {
        if (RefPtr database = m_database)
            database->markAllUnattributedPrivateClickMeasurementAsExpiredForTesting();
    });
}

void PersistentStore::attributePrivateClickMeasurement(WebCore::PCM::SourceSite&& sourceSite, WebCore::PCM::AttributionDestinationSite&& destinationSite, const ApplicationBundleIdentifier& applicationBundleIdentifier, WebCore::PCM::AttributionTriggerData&& attributionTriggerData, WebCore::PrivateClickMeasurement::IsRunningLayoutTest isRunningTest, CompletionHandler<void(std::optional<WebCore::PCM::AttributionSecondsUntilSendData>&&, DebugInfo&&)>&& completionHandler)
{
    postTask([this, protectedThis = Ref { *this }, sourceSite = WTFMove(sourceSite).isolatedCopy(), destinationSite = WTFMove(destinationSite).isolatedCopy(), applicationBundleIdentifier = applicationBundleIdentifier.isolatedCopy(), attributionTriggerData = WTFMove(attributionTriggerData), isRunningTest, completionHandler = WTFMove(completionHandler)] () mutable {
        if (!m_database) {
            return postTaskReply([completionHandler = WTFMove(completionHandler)] () mutable {
                completionHandler(std::nullopt, { });
            });
        }

        auto [seconds, debugInfo] = m_database->attributePrivateClickMeasurement(sourceSite, destinationSite, applicationBundleIdentifier, WTFMove(attributionTriggerData), isRunningTest);

        postTaskReply([seconds = WTFMove(seconds), debugInfo = WTFMove(debugInfo).isolatedCopy(), completionHandler = WTFMove(completionHandler)]() mutable {
            completionHandler(WTFMove(seconds), WTFMove(debugInfo));
        });
    });
}

void PersistentStore::privateClickMeasurementToStringForTesting(CompletionHandler<void(String)>&& completionHandler) const
{
    postTask([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)]() mutable {
        String result;
        if (RefPtr database = m_database)
            result = database->privateClickMeasurementToStringForTesting();
        postTaskReply([result = WTFMove(result).isolatedCopy(), completionHandler = WTFMove(completionHandler)]() mutable {
            completionHandler(result);
        });
    });
}

void PersistentStore::allAttributedPrivateClickMeasurement(CompletionHandler<void(Vector<WebCore::PrivateClickMeasurement>&&)>&& completionHandler)
{
    postTask([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)]() mutable {
        Vector<WebCore::PrivateClickMeasurement> convertedAttributions;
        if (RefPtr database = m_database)
            convertedAttributions = database->allAttributedPrivateClickMeasurement();
        postTaskReply([convertedAttributions = crossThreadCopy(WTFMove(convertedAttributions)), completionHandler = WTFMove(completionHandler)]() mutable {
            completionHandler(WTFMove(convertedAttributions));
        });
    });
}

void PersistentStore::markAttributedPrivateClickMeasurementsAsExpiredForTesting(CompletionHandler<void()>&& completionHandler)
{
    postTask([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)]() mutable {
        if (RefPtr database = m_database)
            database->markAttributedPrivateClickMeasurementsAsExpiredForTesting();
        postTaskReply(WTFMove(completionHandler));
    });
}

void PersistentStore::clearPrivateClickMeasurement(CompletionHandler<void()>&& completionHandler)
{
    postTask([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] () mutable {
        if (RefPtr database = m_database)
            database->clearPrivateClickMeasurement(std::nullopt);
        postTaskReply(WTFMove(completionHandler));
    });
}

void PersistentStore::clearPrivateClickMeasurementForRegistrableDomain(WebCore::RegistrableDomain&& domain, CompletionHandler<void()>&& completionHandler)
{
    postTask([this, protectedThis = Ref { *this }, domain = WTFMove(domain).isolatedCopy(), completionHandler = WTFMove(completionHandler)] () mutable {
        if (RefPtr database = m_database)
            database->clearPrivateClickMeasurement(domain);
        postTaskReply(WTFMove(completionHandler));
    });
}

void PersistentStore::clearExpiredPrivateClickMeasurement()
{
    postTask([this, protectedThis = Ref { *this }]() {
        if (RefPtr database = m_database)
            database->clearExpiredPrivateClickMeasurement();
    });
}

void PersistentStore::clearSentAttribution(WebCore::PrivateClickMeasurement&& attributionToClear, WebCore::PCM::AttributionReportEndpoint attributionReportEndpoint)
{
    postTask([this, protectedThis = Ref { *this }, attributionToClear = WTFMove(attributionToClear).isolatedCopy(), attributionReportEndpoint]() mutable {
        if (RefPtr database = m_database)
            database->clearSentAttribution(WTFMove(attributionToClear), attributionReportEndpoint);
    });
}

void PersistentStore::close(CompletionHandler<void()>&& completionHandler)
{
    postTask([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] () mutable {
        m_database = nullptr;
        postTaskReply(WTFMove(completionHandler));
    });
}

} // namespace WebKit::PCM
