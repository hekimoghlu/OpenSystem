/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#include "ExceptionOr.h"
#include "JSDOMPromiseDeferredForward.h"
#include "PushEncryptionKeyName.h"
#include "PushSubscriptionData.h"
#include "PushSubscriptionJSON.h"

#include <optional>
#include <variant>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
class ArrayBuffer;
}

namespace WebCore {

class PushSubscriptionOptions;
class PushSubscriptionOwner;
class ScriptExecutionContext;
class ServiceWorkerContainer;

class PushSubscription : public RefCounted<PushSubscription> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(PushSubscription, WEBCORE_EXPORT);
public:
    template<typename... Args> static Ref<PushSubscription> create(Args&&... args) { return adoptRef(*new PushSubscription(std::forward<Args>(args)...)); }
    WEBCORE_EXPORT ~PushSubscription();

    WEBCORE_EXPORT const PushSubscriptionData& data() const;

    const String& endpoint() const;
    std::optional<EpochTimeStamp> expirationTime() const;
    PushSubscriptionOptions& options() const;
    const Vector<uint8_t>& clientECDHPublicKey() const;
    const Vector<uint8_t>& sharedAuthenticationSecret() const;

    ExceptionOr<RefPtr<JSC::ArrayBuffer>> getKey(PushEncryptionKeyName) const;
    void unsubscribe(ScriptExecutionContext&, DOMPromiseDeferred<IDLBoolean>&&);

    PushSubscriptionJSON toJSON() const;

private:
    WEBCORE_EXPORT explicit PushSubscription(PushSubscriptionData&&, RefPtr<PushSubscriptionOwner>&& = nullptr);

    PushSubscriptionData m_data;
    RefPtr<PushSubscriptionOwner> m_pushSubscriptionOwner;
    mutable RefPtr<PushSubscriptionOptions> m_options;
};

} // namespace WebCore
