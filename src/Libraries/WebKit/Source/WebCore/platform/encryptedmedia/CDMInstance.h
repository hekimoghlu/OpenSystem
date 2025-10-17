/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
#include <utility>
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/TypeCasts.h>

#if !RELEASE_LOG_DISABLED
namespace WTF {
class Logger;
}
#endif

namespace WebCore {

class CDMInstanceSession;
struct CDMKeySystemConfiguration;
class SharedBuffer;

class CDMInstanceClient : public AbstractRefCountedAndCanMakeWeakPtr<CDMInstanceClient> {
public:
    virtual ~CDMInstanceClient() = default;

    virtual void unrequestedInitializationDataReceived(const String&, Ref<SharedBuffer>&&) = 0;

#if !RELEASE_LOG_DISABLED
    virtual const Logger& logger() const = 0;
    virtual uint64_t logIdentifier() const = 0;
#endif
};

// JavaScript's handle to a CDMInstance, must be used from the
// main-thread only!
class CDMInstance : public RefCounted<CDMInstance> {
public:
    virtual ~CDMInstance() = default;

    virtual void setClient(WeakPtr<CDMInstanceClient>&&) { }
    virtual void clearClient() { }

#if !RELEASE_LOG_DISABLED
    virtual void setLogIdentifier(uint64_t) { }
#endif

    enum class ImplementationType {
        Mock,
        ClearKey,
        FairPlayStreaming,
        Remote,
#if ENABLE(THUNDER)
        Thunder,
#endif
    };
    virtual ImplementationType implementationType() const = 0;

    enum SuccessValue : bool {
        Failed,
        Succeeded,
    };
    using SuccessCallback = CompletionHandler<void(SuccessValue)>;

    enum class AllowDistinctiveIdentifiers : bool {
        No,
        Yes,
    };

    enum class AllowPersistentState : bool {
        No,
        Yes,
    };

    virtual void initializeWithConfiguration(const CDMKeySystemConfiguration&, AllowDistinctiveIdentifiers, AllowPersistentState, SuccessCallback&&) = 0;
    virtual void setServerCertificate(Ref<SharedBuffer>&&, SuccessCallback&&) = 0;
    virtual void setStorageDirectory(const String&) = 0;
    virtual const String& keySystem() const = 0;
    virtual RefPtr<CDMInstanceSession> createSession() = 0;

    enum class HDCPStatus : uint8_t {
        Unknown,
        Valid,
        OutputRestricted,
        OutputDownscaled,
    };
    virtual SuccessValue setHDCPStatus(HDCPStatus) { return Failed; }
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CDM_INSTANCE(ToValueTypeName, ImplementationTypeName) \
SPECIALIZE_TYPE_TRAITS_BEGIN(ToValueTypeName) \
static bool isType(const WebCore::CDMInstance& instance) { return instance.implementationType() == ImplementationTypeName; } \
SPECIALIZE_TYPE_TRAITS_END()

#endif
