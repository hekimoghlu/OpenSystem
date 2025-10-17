/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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

#include "EpochTimeStamp.h"
#include "PushSubscriptionIdentifier.h"
#include "SQLiteDatabase.h"
#include "SQLiteStatement.h"
#include "SQLiteStatementAutoResetScope.h"
#include <span>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UUID.h>
#include <wtf/UniqueRef.h>
#include <wtf/Vector.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

struct PushRecord {
    Markable<PushSubscriptionIdentifier> identifier { };
    PushSubscriptionSetIdentifier subscriptionSetIdentifier;
    String securityOrigin;
    String scope;
    String endpoint;
    String topic;
    Vector<uint8_t> serverVAPIDPublicKey;
    Vector<uint8_t> clientPublicKey;
    Vector<uint8_t> clientPrivateKey;
    Vector<uint8_t> sharedAuthSecret;
    std::optional<EpochTimeStamp> expirationTime { };

    WEBCORE_EXPORT PushRecord isolatedCopy() const &;
    WEBCORE_EXPORT PushRecord isolatedCopy() &&;
};

struct RemovedPushRecord {
    PushSubscriptionIdentifier identifier;
    String topic;
    Vector<uint8_t> serverVAPIDPublicKey;

    WEBCORE_EXPORT RemovedPushRecord isolatedCopy() const &;
    WEBCORE_EXPORT RemovedPushRecord isolatedCopy() &&;
};

struct PushSubscriptionSetRecord {
    PushSubscriptionSetIdentifier identifier;
    String securityOrigin;
    bool enabled;

    WEBCORE_EXPORT PushSubscriptionSetRecord isolatedCopy() const &;
    WEBCORE_EXPORT PushSubscriptionSetRecord isolatedCopy() &&;
};

struct PushTopics {
    Vector<String> enabledTopics;
    Vector<String> ignoredTopics;

    WEBCORE_EXPORT PushTopics isolatedCopy() const &;
    WEBCORE_EXPORT PushTopics isolatedCopy() &&;
};

class PushDatabase : public RefCountedAndCanMakeWeakPtr<PushDatabase> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(PushDatabase, WEBCORE_EXPORT);
public:
    using CreationHandler = CompletionHandler<void(RefPtr<PushDatabase>&&)>;

    WEBCORE_EXPORT static void create(const String& path, CreationHandler&&);
    WEBCORE_EXPORT ~PushDatabase();

    enum class PublicTokenChanged : bool { No, Yes };
    WEBCORE_EXPORT void updatePublicToken(std::span<const uint8_t>, CompletionHandler<void(PublicTokenChanged)>&&);
    WEBCORE_EXPORT void getPublicToken(CompletionHandler<void(Vector<uint8_t>&&)>&&);

    WEBCORE_EXPORT void insertRecord(const PushRecord&, CompletionHandler<void(std::optional<PushRecord>&&)>&&);
    WEBCORE_EXPORT void removeRecordByIdentifier(PushSubscriptionIdentifier, CompletionHandler<void(bool)>&&);
    WEBCORE_EXPORT void getRecordByTopic(const String& topic, CompletionHandler<void(std::optional<PushRecord>&&)>&&);
    WEBCORE_EXPORT void getRecordBySubscriptionSetAndScope(const PushSubscriptionSetIdentifier&, const String& scope, CompletionHandler<void(std::optional<PushRecord>&&)>&&);
    WEBCORE_EXPORT void getIdentifiers(CompletionHandler<void(HashSet<PushSubscriptionIdentifier>&&)>&&);
    WEBCORE_EXPORT void getPushSubscriptionSetRecords(CompletionHandler<void(Vector<PushSubscriptionSetRecord>&&)>&&);
    WEBCORE_EXPORT void getTopics(CompletionHandler<void(PushTopics&&)>&&);

    WEBCORE_EXPORT void incrementSilentPushCount(const PushSubscriptionSetIdentifier&, const String& securityOrigin, CompletionHandler<void(unsigned)>&&);

    WEBCORE_EXPORT void removeRecordsBySubscriptionSet(const PushSubscriptionSetIdentifier&, CompletionHandler<void(Vector<RemovedPushRecord>&&)>&&);
    WEBCORE_EXPORT void removeRecordsBySubscriptionSetAndSecurityOrigin(const PushSubscriptionSetIdentifier&, const String& securityOrigin, CompletionHandler<void(Vector<RemovedPushRecord>&&)>&&);
    WEBCORE_EXPORT void removeRecordsByBundleIdentifierAndDataStore(const String& bundleIdentifier, const std::optional<WTF::UUID>& dataStoreIdentifier, CompletionHandler<void(Vector<RemovedPushRecord>&&)>&&);

    WEBCORE_EXPORT void setPushesEnabled(const PushSubscriptionSetIdentifier&, bool, CompletionHandler<void(bool recordsChanged)>&&);
    WEBCORE_EXPORT void setPushesEnabledForOrigin(const PushSubscriptionSetIdentifier&, const String& securityOrigin, bool, CompletionHandler<void(bool recordsChanged)>&&);

private:
    PushDatabase(Ref<WorkQueue>&&, UniqueRef<WebCore::SQLiteDatabase>&&);

    WebCore::SQLiteStatementAutoResetScope cachedStatementOnQueue(ASCIILiteral query);
    template<typename... Args> WebCore::SQLiteStatementAutoResetScope bindStatementOnQueue(ASCIILiteral query, Args&&...);

    void dispatchOnWorkQueue(Function<void()>&&);

    Ref<WorkQueue> m_queue;
    UniqueRef<WebCore::SQLiteDatabase> m_db;
    HashMap<ASCIILiteral, UniqueRef<WebCore::SQLiteStatement>> m_statements;
};

} // namespace WebCore
