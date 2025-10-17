/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
#include "RemoteCDMInstanceSession.h"

#if ENABLE(GPU_PROCESS) && ENABLE(ENCRYPTED_MEDIA)

#include "GPUProcessConnection.h"
#include "RemoteCDMInstanceSessionProxyMessages.h"
#include <WebCore/SharedBuffer.h>
#include <wtf/Ref.h>

namespace WebKit {

using namespace WebCore;

Ref<RemoteCDMInstanceSession> RemoteCDMInstanceSession::create(WeakPtr<RemoteCDMFactory>&& factory, RemoteCDMInstanceSessionIdentifier&& identifier)
{
    return adoptRef(*new RemoteCDMInstanceSession(WTFMove(factory), WTFMove(identifier)));
}

RemoteCDMInstanceSession::RemoteCDMInstanceSession(WeakPtr<RemoteCDMFactory>&& factory, RemoteCDMInstanceSessionIdentifier&& identifier)
    : m_factory(WTFMove(factory))
    , m_identifier(WTFMove(identifier))
{
}

RemoteCDMInstanceSession::~RemoteCDMInstanceSession()
{
    protectedFactory()->removeSession(m_identifier);
}

RefPtr<RemoteCDMFactory> RemoteCDMInstanceSession::protectedFactory() const
{
    return m_factory.get();
}

#if !RELEASE_LOG_DISABLED
void RemoteCDMInstanceSession::setLogIdentifier(uint64_t logIdentifier)
{
    protectedFactory()->gpuProcessConnection().connection().send(Messages::RemoteCDMInstanceSessionProxy::SetLogIdentifier(reinterpret_cast<uint64_t>(logIdentifier)), m_identifier);
}
#endif

void RemoteCDMInstanceSession::requestLicense(LicenseType type, KeyGroupingStrategy keyGroupingStrategy, const AtomString& initDataType, Ref<SharedBuffer>&& initData, LicenseCallback&& callback)
{
    RefPtr factory = m_factory.get();
    if (!factory) {
        callback(SharedBuffer::create(), emptyString(), false, Failed);
        return;
    }

    factory->gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteCDMInstanceSessionProxy::RequestLicense(type, keyGroupingStrategy, initDataType, WTFMove(initData)), [callback = WTFMove(callback)] (RefPtr<SharedBuffer>&& message, const String& sessionId, bool needsIndividualization, bool succeeded) mutable {
        if (!message) {
            callback(SharedBuffer::create(), emptyString(), false, Failed);
            return;
        }
        callback(message.releaseNonNull(), sessionId, needsIndividualization, succeeded ? Succeeded : Failed);
    }, m_identifier);
}

void RemoteCDMInstanceSession::updateLicense(const String& sessionId, LicenseType type, Ref<SharedBuffer>&& response, LicenseUpdateCallback&& callback)
{
    RefPtr factory = m_factory.get();
    if (!factory) {
        callback(false, std::nullopt, std::nullopt, std::nullopt, Failed);
        return;
    }

    factory->gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteCDMInstanceSessionProxy::UpdateLicense(sessionId, type, WTFMove(response)), [callback = WTFMove(callback)] (bool sessionWasClosed, std::optional<KeyStatusVector>&& changedKeys, std::optional<double>&& changedExpiration, std::optional<Message>&& message, bool succeeded) mutable {
        callback(sessionWasClosed, WTFMove(changedKeys), WTFMove(changedExpiration), WTFMove(message), succeeded ? Succeeded : Failed);
    }, m_identifier);
}

void RemoteCDMInstanceSession::loadSession(LicenseType type, const String& sessionId, const String& origin, LoadSessionCallback&& callback)
{
    RefPtr factory = m_factory.get();
    if (!factory) {
        callback(std::nullopt, std::nullopt, std::nullopt, Failed, SessionLoadFailure::Other);
        return;
    }

    factory->gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteCDMInstanceSessionProxy::LoadSession(type, sessionId, origin), [callback = WTFMove(callback)] (std::optional<KeyStatusVector>&& changedKeys, std::optional<double>&& changedExpiration, std::optional<Message>&& message, bool succeeded, SessionLoadFailure loadFailure) mutable {
        callback(WTFMove(changedKeys), WTFMove(changedExpiration), WTFMove(message), succeeded ? Succeeded : Failed, loadFailure);
    }, m_identifier);
}

void RemoteCDMInstanceSession::closeSession(const String& sessionId, CloseSessionCallback&& callback)
{
    RefPtr factory = m_factory.get();
    if (!factory) {
        callback();
        return;
    }

    factory->gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteCDMInstanceSessionProxy::CloseSession(sessionId), [callback = WTFMove(callback)] () mutable {
        callback();
    }, m_identifier);
}

void RemoteCDMInstanceSession::removeSessionData(const String& sessionId, LicenseType type, RemoveSessionDataCallback&& callback)
{
    RefPtr factory = m_factory.get();
    if (!factory) {
        callback({ }, nullptr, Failed);
        return;
    }

    factory->gpuProcessConnection().connection().sendWithAsyncReply(Messages::RemoteCDMInstanceSessionProxy::RemoveSessionData(sessionId, type), [callback = WTFMove(callback)] (KeyStatusVector&& changedKeys, RefPtr<SharedBuffer>&& message, bool succeeded) mutable {
        callback(WTFMove(changedKeys), WTFMove(message), succeeded ? Succeeded : Failed);
    }, m_identifier);
}

void RemoteCDMInstanceSession::storeRecordOfKeyUsage(const String& sessionId)
{
    if (RefPtr factory = m_factory.get())
        factory->gpuProcessConnection().connection().send(Messages::RemoteCDMInstanceSessionProxy::StoreRecordOfKeyUsage(sessionId), m_identifier);
}

void RemoteCDMInstanceSession::updateKeyStatuses(KeyStatusVector&& keyStatuses)
{
    if (m_client)
        m_client->updateKeyStatuses(WTFMove(keyStatuses));
}

void RemoteCDMInstanceSession::sendMessage(WebCore::CDMMessageType type, RefPtr<SharedBuffer>&& message)
{
    if (m_client && message)
        m_client->sendMessage(type, message.releaseNonNull());
}

void RemoteCDMInstanceSession::sessionIdChanged(const String& sessionId)
{
    if (m_client)
        m_client->sessionIdChanged(sessionId);
}

}

#endif
