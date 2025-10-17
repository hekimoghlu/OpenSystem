/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 29, 2024.
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

#include "DatabaseUtilities.h"
#include <WebCore/PrivateClickMeasurement.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebKit {
namespace PCM {
class Database;
}
}

namespace WebKit::PCM {

struct DebugInfo;

// This is created, used, and destroyed on the Store's queue.
class Database : public DatabaseUtilities, public RefCountedAndCanMakeWeakPtr<Database> {
    WTF_MAKE_TZONE_ALLOCATED(Database);
public:
    static Ref<Database> create(const String& storageDirectory);

    virtual ~Database();

    using ApplicationBundleIdentifier = String;

    static void interruptAllDatabases();

    void insertPrivateClickMeasurement(WebCore::PrivateClickMeasurement&&, PrivateClickMeasurementAttributionType);
    std::pair<std::optional<WebCore::PCM::AttributionSecondsUntilSendData>, DebugInfo> attributePrivateClickMeasurement(const WebCore::PCM::SourceSite&, const WebCore::PCM::AttributionDestinationSite&, const ApplicationBundleIdentifier&, WebCore::PCM::AttributionTriggerData&&, WebCore::PrivateClickMeasurement::IsRunningLayoutTest);
    Vector<WebCore::PrivateClickMeasurement> allAttributedPrivateClickMeasurement();
    void clearPrivateClickMeasurement(std::optional<WebCore::RegistrableDomain>);
    void clearExpiredPrivateClickMeasurement();
    void clearSentAttribution(WebCore::PrivateClickMeasurement&&, WebCore::PCM::AttributionReportEndpoint);

    String privateClickMeasurementToStringForTesting() const;
    void markAllUnattributedPrivateClickMeasurementAsExpiredForTesting();
    void markAttributedPrivateClickMeasurementsAsExpiredForTesting();

private:
    using UnattributedPrivateClickMeasurement = WebCore::PrivateClickMeasurement;
    using AttributedPrivateClickMeasurement = WebCore::PrivateClickMeasurement;
    using DomainID = unsigned;
    using SourceDomainID = unsigned;
    using DestinationDomainID = unsigned;
    using SourceEarliestTimeToSend = double;
    using DestinationEarliestTimeToSend = double;

    Database(const String& storageDirectory);
    bool createSchema() final;
    void destroyStatements() final;
    std::pair<std::optional<UnattributedPrivateClickMeasurement>, std::optional<AttributedPrivateClickMeasurement>> findPrivateClickMeasurement(const WebCore::PCM::SourceSite&, const WebCore::PCM::AttributionDestinationSite&, const ApplicationBundleIdentifier&);
    void removeUnattributed(WebCore::PrivateClickMeasurement&);
    String attributionToStringForTesting(const WebCore::PrivateClickMeasurement&) const;
    void markReportAsSentToDestination(SourceDomainID, DestinationDomainID, const ApplicationBundleIdentifier&);
    void markReportAsSentToSource(SourceDomainID, DestinationDomainID, const ApplicationBundleIdentifier&);
    std::pair<std::optional<SourceEarliestTimeToSend>, std::optional<DestinationEarliestTimeToSend>> earliestTimesToSend(const WebCore::PrivateClickMeasurement&);
    std::optional<DomainID> ensureDomainID(const WebCore::RegistrableDomain&);
    std::optional<DomainID> domainID(const WebCore::RegistrableDomain&);
    String getDomainStringFromDomainID(DomainID) const final;

    void addDestinationTokenColumnsIfNecessary();
    bool needsUpdatedSchema() final { return false; };
    bool createUniqueIndices() final;
    const MemoryCompactLookupOnlyRobinHoodHashMap<String, TableAndIndexPair>& expectedTableAndIndexQueries() final;
    std::span<const ASCIILiteral> sortedTables() final;

    using Statement = std::unique_ptr<WebCore::SQLiteStatement>;
    mutable Statement m_setUnattributedPrivateClickMeasurementAsExpiredStatement;
    mutable Statement m_findUnattributedStatement;
    mutable Statement m_findAttributedStatement;
    mutable Statement m_removeUnattributedStatement;
    mutable Statement m_allAttributedPrivateClickMeasurementStatement;
    mutable Statement m_allUnattributedPrivateClickMeasurementAttributionsStatement;
    mutable Statement m_clearAllPrivateClickMeasurementStatement;
    mutable Statement m_clearExpiredPrivateClickMeasurementStatement;
    mutable Statement m_earliestTimesToSendStatement;
    mutable Statement m_markReportAsSentToSourceStatement;
    mutable Statement m_markReportAsSentToDestinationStatement;
    mutable Statement m_domainIDFromStringStatement;
    mutable Statement m_domainStringFromDomainIDStatement;
    mutable Statement m_insertObservedDomainStatement;
};

} // namespace WebKit::PCM
