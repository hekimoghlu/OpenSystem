/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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
#include "PushSubscription.h"

#include "EventLoop.h"
#include "Exception.h"
#include "JSDOMPromiseDeferred.h"
#include "PushSubscriptionOptions.h"
#include "PushSubscriptionOwner.h"
#include "ScriptExecutionContext.h"
#include "ServiceWorkerContainer.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/Base64.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PushSubscription);

PushSubscription::PushSubscription(PushSubscriptionData&& data, RefPtr<PushSubscriptionOwner>&& owner)
    : m_data(WTFMove(data))
    , m_pushSubscriptionOwner(WTFMove(owner))
{
}

PushSubscription::~PushSubscription() = default;

const PushSubscriptionData& PushSubscription::data() const
{
    return m_data;
}

const String& PushSubscription::endpoint() const
{
    return m_data.endpoint;
}

std::optional<EpochTimeStamp> PushSubscription::expirationTime() const
{
    return m_data.expirationTime;
}

PushSubscriptionOptions& PushSubscription::options() const
{
    if (!m_options) {
        auto key = m_data.serverVAPIDPublicKey;
        m_options = PushSubscriptionOptions::create(WTFMove(key));
    }

    return *m_options;
}

const Vector<uint8_t>& PushSubscription::clientECDHPublicKey() const
{
    return m_data.clientECDHPublicKey;
}

const Vector<uint8_t>& PushSubscription::sharedAuthenticationSecret() const
{
    return m_data.sharedAuthenticationSecret;
}

ExceptionOr<RefPtr<JSC::ArrayBuffer>> PushSubscription::getKey(PushEncryptionKeyName name) const
{
    auto& source = [&]() -> const Vector<uint8_t>& {
        switch (name) {
        case PushEncryptionKeyName::P256dh:
            return clientECDHPublicKey();
        case PushEncryptionKeyName::Auth:
            return sharedAuthenticationSecret();
        }
    }();

    auto buffer = ArrayBuffer::tryCreate(source);
    if (!buffer)
        return Exception { ExceptionCode::OutOfMemoryError };
    return buffer;
}

void PushSubscription::unsubscribe(ScriptExecutionContext& scriptExecutionContext, DOMPromiseDeferred<IDLBoolean>&& promise)
{
    scriptExecutionContext.eventLoop().queueTask(TaskSource::Networking, [this, protectedThis = Ref { *this }, pushSubscriptionIdentifier = m_data.identifier, promise = WTFMove(promise)]() mutable {
        if (!m_pushSubscriptionOwner) {
            promise.resolve(false);
            return;
        }

        m_pushSubscriptionOwner->unsubscribeFromPushService(pushSubscriptionIdentifier, WTFMove(promise));
    });
}

PushSubscriptionJSON PushSubscription::toJSON() const
{
    return PushSubscriptionJSON {
        endpoint(),
        expirationTime(),
        Vector<KeyValuePair<String, String>> {
            { "p256dh"_s, base64URLEncodeToString(clientECDHPublicKey()) },
            { "auth"_s, base64URLEncodeToString(sharedAuthenticationSecret()) }
        }
    };
}

} // namespace WebCore
