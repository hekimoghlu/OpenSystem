/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 25, 2025.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "CDMKeyStatus.h"
#include "CDMMessageType.h"
#include "CDMSessionType.h"
#include <wtf/CompletionHandler.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

#if !RELEASE_LOG_DISABLED
namespace WTF {
class Logger;
}
#endif

namespace WebCore {
class CDMInstanceSessionClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::CDMInstanceSessionClient> : std::true_type { };
}

namespace WebCore {

class SharedBuffer;

enum class CDMKeyGroupingStrategy : bool;

enum class CDMInstanceSessionLoadFailure : uint8_t {
    None,
    NoSessionData,
    MismatchedSessionType,
    QuotaExceeded,
    Other,
};

class CDMInstanceSessionClient : public CanMakeWeakPtr<CDMInstanceSessionClient> {
public:
    virtual ~CDMInstanceSessionClient() = default;

    using KeyStatus = CDMKeyStatus;
    using KeyStatusVector = Vector<std::pair<Ref<SharedBuffer>, KeyStatus>>;
    virtual void updateKeyStatuses(KeyStatusVector&&) = 0;
    virtual void sendMessage(CDMMessageType, Ref<SharedBuffer>&& message) = 0;
    virtual void sessionIdChanged(const String&) = 0;
    using PlatformDisplayID = uint32_t;
    virtual PlatformDisplayID displayID() = 0;
};

class CDMInstanceSession : public RefCounted<CDMInstanceSession> {
public:
    virtual ~CDMInstanceSession() = default;

    using KeyGroupingStrategy = CDMKeyGroupingStrategy;
    using KeyStatus = CDMKeyStatus;
    using LicenseType = CDMSessionType;
    using MessageType = CDMMessageType;

#if !RELEASE_LOG_DISABLED
    virtual void setLogIdentifier(uint64_t) { }
#endif

    virtual void setClient(WeakPtr<CDMInstanceSessionClient>&&) { }
    virtual void clearClient() { }

    enum SuccessValue {
        Failed,
        Succeeded,
    };

    using LicenseCallback = CompletionHandler<void(Ref<SharedBuffer>&& message, const String& sessionId, bool needsIndividualization, SuccessValue succeeded)>;
    virtual void requestLicense(LicenseType, KeyGroupingStrategy, const AtomString& initDataType, Ref<SharedBuffer>&& initData, LicenseCallback&&) = 0;

    using KeyStatusVector = CDMInstanceSessionClient::KeyStatusVector;
    using Message = std::pair<MessageType, Ref<SharedBuffer>>;
    using LicenseUpdateCallback = CompletionHandler<void(bool sessionWasClosed, std::optional<KeyStatusVector>&& changedKeys, std::optional<double>&& changedExpiration, std::optional<Message>&& message, SuccessValue succeeded)>;
    virtual void updateLicense(const String& sessionId, LicenseType, Ref<SharedBuffer>&& response, LicenseUpdateCallback&&) = 0;

    using SessionLoadFailure = CDMInstanceSessionLoadFailure;

    using LoadSessionCallback = CompletionHandler<void(std::optional<KeyStatusVector>&&, std::optional<double>&&, std::optional<Message>&&, SuccessValue, SessionLoadFailure)>;
    virtual void loadSession(LicenseType, const String& sessionId, const String& origin, LoadSessionCallback&&) = 0;

    using CloseSessionCallback = CompletionHandler<void()>;
    virtual void closeSession(const String& sessionId, CloseSessionCallback&&) = 0;

    using RemoveSessionDataCallback = CompletionHandler<void(KeyStatusVector&&, RefPtr<SharedBuffer>&&, SuccessValue)>;
    virtual void removeSessionData(const String& sessionId, LicenseType, RemoveSessionDataCallback&&) = 0;

    virtual void storeRecordOfKeyUsage(const String& sessionId) = 0;

    using PlatformDisplayID = uint32_t;
    virtual void displayChanged(PlatformDisplayID) { }
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CDM_INSTANCE(ToValueTypeName, ImplementationTypeName) \
SPECIALIZE_TYPE_TRAITS_BEGIN(ToValueTypeName) \
static bool isType(const WebCore::CDMInstance& instance) { return instance.implementationType() == ImplementationTypeName; } \
SPECIALIZE_TYPE_TRAITS_END()

#endif
