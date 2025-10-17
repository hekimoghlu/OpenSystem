/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 23, 2025.
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
#include "RemoteLegacyCDMSession.h"

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "GPUProcessConnection.h"
#include "RemoteLegacyCDMFactory.h"
#include "RemoteLegacyCDMSessionProxyMessages.h"
#include <JavaScriptCore/GenericTypedArrayViewInlines.h>
#include <JavaScriptCore/JSGenericTypedArrayViewInlines.h>
#include <JavaScriptCore/TypedArrayType.h>
#include <WebCore/SharedBuffer.h>

namespace WebKit {

using namespace WebCore;

static RefPtr<ArrayBuffer> convertToArrayBuffer(RefPtr<const SharedBuffer>&& buffer)
{
    if (buffer)
        return buffer->tryCreateArrayBuffer();
    return nullptr;
}

static RefPtr<Uint8Array> convertToUint8Array(RefPtr<const SharedBuffer>&& buffer)
{
    auto arrayBuffer = convertToArrayBuffer(WTFMove(buffer));
    if (!arrayBuffer)
        return nullptr;

    size_t sizeInBytes = arrayBuffer->byteLength();
    return Uint8Array::create(WTFMove(arrayBuffer), 0, sizeInBytes);
}

template <typename T>
static RefPtr<SharedBuffer> convertToSharedBuffer(T array)
{
    if (!array)
        return nullptr;
    return SharedBuffer::create(array->span());
}

RefPtr<RemoteLegacyCDMSession> RemoteLegacyCDMSession::create(RemoteLegacyCDMFactory& factory, RemoteLegacyCDMSessionIdentifier&& identifier, LegacyCDMSessionClient& client)
{
    RefPtr session = adoptRef(new RemoteLegacyCDMSession(factory, WTFMove(identifier), client));
    if (session->m_factory)
        session->m_factory->addSession(identifier, *session);
    return session;
}

RemoteLegacyCDMSession::RemoteLegacyCDMSession(RemoteLegacyCDMFactory& factory, RemoteLegacyCDMSessionIdentifier&& identifier, LegacyCDMSessionClient& client)
    : m_factory(WTFMove(factory))
    , m_identifier(WTFMove(identifier))
    , m_client(client)
{
}

RemoteLegacyCDMSession::~RemoteLegacyCDMSession()
{
    ASSERT(!m_factory);
}

void RemoteLegacyCDMSession::invalidate()
{
    if (RefPtr factory = m_factory.get()) {
        factory->removeSession(m_identifier);
        m_factory = nullptr;
    }
}

RefPtr<Uint8Array> RemoteLegacyCDMSession::generateKeyRequest(const String& mimeType, Uint8Array* initData, String& destinationURL, unsigned short& errorCode, uint32_t& systemCode)
{
    if (!m_factory || !initData)
        return nullptr;

    auto ipcInitData = convertToSharedBuffer(initData);
    auto sendResult = m_factory->gpuProcessConnection().connection().sendSync(Messages::RemoteLegacyCDMSessionProxy::GenerateKeyRequest(mimeType, ipcInitData), m_identifier);

    RefPtr<SharedBuffer> ipcNextMessage;
    if (sendResult.succeeded())
        std::tie(ipcNextMessage, destinationURL, errorCode, systemCode) = sendResult.takeReply();

    if (!ipcNextMessage)
        return nullptr;

    return convertToUint8Array(WTFMove(ipcNextMessage));
}

void RemoteLegacyCDMSession::releaseKeys()
{
    if (!m_factory)
        return;

    m_factory->gpuProcessConnection().connection().send(Messages::RemoteLegacyCDMSessionProxy::ReleaseKeys(), m_identifier);
    m_cachedKeyCache.clear();
}

bool RemoteLegacyCDMSession::update(Uint8Array* keyData, RefPtr<Uint8Array>& nextMessage, unsigned short& errorCode, uint32_t& systemCode)
{
    if (!m_factory || !keyData)
        return false;

    auto ipcKeyData = convertToSharedBuffer(keyData);
    auto sendResult = m_factory->gpuProcessConnection().connection().sendSync(Messages::RemoteLegacyCDMSessionProxy::Update(ipcKeyData), m_identifier);

    bool succeeded { false };
    RefPtr<SharedBuffer> ipcNextMessage;
    if (sendResult.succeeded())
        std::tie(succeeded, ipcNextMessage, errorCode, systemCode) = sendResult.takeReply();

    if (ipcNextMessage)
        nextMessage = convertToUint8Array(WTFMove(ipcNextMessage));

    return succeeded;
}

RefPtr<ArrayBuffer> RemoteLegacyCDMSession::cachedKeyForKeyID(const String& keyId) const
{
    if (!m_factory)
        return nullptr;

    auto foundInCache = m_cachedKeyCache.find(keyId);
    if (foundInCache != m_cachedKeyCache.end())
        return foundInCache->value;

    auto sendResult = m_factory->gpuProcessConnection().connection().sendSync(Messages::RemoteLegacyCDMSessionProxy::CachedKeyForKeyID(keyId), m_identifier);
    auto [ipcKey] = sendResult.takeReplyOr(nullptr);

    if (!ipcKey)
        return nullptr;

    auto ipcKeyBuffer = convertToArrayBuffer(WTFMove(ipcKey));
    m_cachedKeyCache.set(keyId, ipcKeyBuffer);
    return ipcKeyBuffer;
}

void RemoteLegacyCDMSession::sendMessage(RefPtr<SharedBuffer>&& message, const String& destinationURL)
{
    if (!m_client)
        return;

    if (!message) {
        m_client->sendMessage(nullptr, destinationURL);
        return;
    }

    m_client->sendMessage(convertToUint8Array(WTFMove(message)).get(), destinationURL);
}

void RemoteLegacyCDMSession::sendError(WebCore::LegacyCDMSessionClient::MediaKeyErrorCode errorCode, uint32_t systemCode)
{
    if (m_client)
        m_client->sendError(errorCode, systemCode);
}

}

#endif
