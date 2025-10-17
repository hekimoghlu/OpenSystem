/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

#include "PrivateClickMeasurementStore.h"

namespace WebKit {

enum class PrivateClickMeasurementAttributionType : bool;

namespace PCM {

class Database;
struct DebugInfo;

class PersistentStore : public Store {
public:
    static Ref<Store> create(const String& databaseDirectory)
    {
        return adoptRef(*new PersistentStore(databaseDirectory));
    }

    ~PersistentStore();

    using ApplicationBundleIdentifier = String;

    static void prepareForProcessToSuspend(CompletionHandler<void()>&&);
    static void processDidResume();

    void insertPrivateClickMeasurement(WebCore::PrivateClickMeasurement&&, WebKit::PrivateClickMeasurementAttributionType, CompletionHandler<void()>&&) final;
    void attributePrivateClickMeasurement(WebCore::PCM::SourceSite&&, WebCore::PCM::AttributionDestinationSite&&, const ApplicationBundleIdentifier&, WebCore::PCM::AttributionTriggerData&&, WebCore::PrivateClickMeasurement::IsRunningLayoutTest, CompletionHandler<void(std::optional<WebCore::PCM::AttributionSecondsUntilSendData>&&, DebugInfo&&)>&&) final;

    void privateClickMeasurementToStringForTesting(CompletionHandler<void(String)>&&) const final;
    void markAllUnattributedPrivateClickMeasurementAsExpiredForTesting() final;
    void markAttributedPrivateClickMeasurementsAsExpiredForTesting(CompletionHandler<void()>&&) final;

    void allAttributedPrivateClickMeasurement(CompletionHandler<void(Vector<WebCore::PrivateClickMeasurement>&&)>&&) final;
    void clearExpiredPrivateClickMeasurement() final;
    void clearPrivateClickMeasurement(CompletionHandler<void()>&&) final;
    void clearPrivateClickMeasurementForRegistrableDomain(WebCore::RegistrableDomain&&, CompletionHandler<void()>&&) final;
    void clearSentAttribution(WebCore::PrivateClickMeasurement&& attributionToClear, WebCore::PCM::AttributionReportEndpoint) final;

    void close(CompletionHandler<void()>&&) final;

private:
    PersistentStore(const String& databaseDirectory);

    void postTask(Function<void()>&&) const;
    void postTaskReply(Function<void()>&&) const;

    RefPtr<Database> m_database;
    Ref<SuspendableWorkQueue> m_queue;
};

} // namespace PCM

} // namespace WebKit
