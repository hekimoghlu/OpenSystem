/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
#include "RemoteLegacyCDMSessionProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include "GPUConnectionToWebProcess.h"
#include "Logging.h"
#include "RemoteLegacyCDMFactoryProxy.h"
#include "RemoteLegacyCDMSessionMessages.h"
#include <JavaScriptCore/GenericTypedArrayViewInlines.h>
#include <JavaScriptCore/TypedArrayAdaptors.h>
#include <WebCore/LegacyCDM.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/LoggerHelper.h>

namespace WebKit {

using namespace WebCore;

Ref<RemoteLegacyCDMSessionProxy> RemoteLegacyCDMSessionProxy::create(RemoteLegacyCDMFactoryProxy& factory, uint64_t logIdentifier, RemoteLegacyCDMSessionIdentifier sessionIdentifier, WebCore::LegacyCDM& cdm)
{
    return adoptRef(*new RemoteLegacyCDMSessionProxy(factory, logIdentifier, sessionIdentifier, cdm));
}

RemoteLegacyCDMSessionProxy::RemoteLegacyCDMSessionProxy(RemoteLegacyCDMFactoryProxy& factory, uint64_t parentLogIdentifier, RemoteLegacyCDMSessionIdentifier sessionIdentifier, WebCore::LegacyCDM& cdm)
    : m_factory(factory)
#if !RELEASE_LOG_DISABLED
    , m_logger(factory.logger())
    , m_logIdentifier(parentLogIdentifier)
#endif
    , m_identifier(sessionIdentifier)
    , m_session(cdm.createSession(*this))
{
    if (!m_session)
        ERROR_LOG(LOGIDENTIFIER, "could not create CDM session.");
}

RemoteLegacyCDMSessionProxy::~RemoteLegacyCDMSessionProxy() = default;

void RemoteLegacyCDMSessionProxy::invalidate()
{
    m_factory = nullptr;
}

static RefPtr<Uint8Array> convertToUint8Array(RefPtr<SharedBuffer>&& buffer)
{
    if (!buffer)
        return nullptr;

    auto arrayBuffer = buffer->tryCreateArrayBuffer();
    if (!arrayBuffer)
        return nullptr;
    return Uint8Array::create(arrayBuffer.releaseNonNull(), 0, buffer->size());
}

template <typename T>
static RefPtr<WebCore::SharedBuffer> convertToOptionalSharedBuffer(T array)
{
    if (!array)
        return nullptr;
    return SharedBuffer::create(array->span());
}

void RemoteLegacyCDMSessionProxy::setPlayer(WeakPtr<RemoteMediaPlayerProxy> player)
{
    m_player = WTFMove(player);
}

void RemoteLegacyCDMSessionProxy::generateKeyRequest(const String& mimeType, RefPtr<SharedBuffer>&& initData, GenerateKeyCallback&& completion)
{
    RefPtr session = m_session;
    if (!session) {
        completion({ }, emptyString(), 0, 0);
        return;
    }
    
    auto initDataArray = convertToUint8Array(WTFMove(initData));
    if (!initDataArray) {
        completion({ }, emptyString(), 0, 0);
        return;
    }

    String destinationURL;
    unsigned short errorCode { 0 };
    uint32_t systemCode { 0 };

    auto keyRequest = session->generateKeyRequest(mimeType, initDataArray.get(), destinationURL, errorCode, systemCode);

    destinationURL = "this is a test string"_s;

    completion(convertToOptionalSharedBuffer(keyRequest), destinationURL, errorCode, systemCode);
}

void RemoteLegacyCDMSessionProxy::releaseKeys()
{
    if (RefPtr session = m_session)
        session->releaseKeys();
}

void RemoteLegacyCDMSessionProxy::update(RefPtr<SharedBuffer>&& update, UpdateCallback&& completion)
{
    RefPtr session = m_session;
    if (!session) {
        completion(false, nullptr, 0, 0);
        return;
    }
    
    auto updateArray = convertToUint8Array(WTFMove(update));
    if (!updateArray) {
        completion(false, nullptr, 0, 0);
        return;
    }

    RefPtr<Uint8Array> nextMessage;
    unsigned short errorCode { 0 };
    uint32_t systemCode { 0 };

    bool succeeded = session->update(updateArray.get(), nextMessage, errorCode, systemCode);

    completion(succeeded, convertToOptionalSharedBuffer(nextMessage), errorCode, systemCode);
}

RefPtr<ArrayBuffer> RemoteLegacyCDMSessionProxy::getCachedKeyForKeyId(const String& keyId)
{
    RefPtr session = m_session;
    if (!session)
        return nullptr;
    
    return session->cachedKeyForKeyID(keyId);
}

void RemoteLegacyCDMSessionProxy::cachedKeyForKeyID(String keyId, CachedKeyForKeyIDCallback&& completion)
{
    completion(convertToOptionalSharedBuffer(getCachedKeyForKeyId(keyId)));
}

void RemoteLegacyCDMSessionProxy::sendMessage(Uint8Array* message, String destinationURL)
{
    RefPtr factory = m_factory.get();
    if (!factory)
        return;

    RefPtr gpuConnectionToWebProcess = factory->gpuConnectionToWebProcess();
    if (!gpuConnectionToWebProcess)
        return;

    gpuConnectionToWebProcess->protectedConnection()->send(Messages::RemoteLegacyCDMSession::SendMessage(convertToOptionalSharedBuffer(message), destinationURL), m_identifier);
}

void RemoteLegacyCDMSessionProxy::sendError(MediaKeyErrorCode errorCode, uint32_t systemCode)
{
    RefPtr factory = m_factory.get();
    if (!factory)
        return;

    RefPtr gpuConnectionToWebProcess = factory->gpuConnectionToWebProcess();
    if (!gpuConnectionToWebProcess)
        return;

    gpuConnectionToWebProcess->protectedConnection()->send(Messages::RemoteLegacyCDMSession::SendError(errorCode, systemCode), m_identifier);
}

String RemoteLegacyCDMSessionProxy::mediaKeysStorageDirectory() const
{
    RefPtr factory = m_factory.get();
    if (!factory)
        return emptyString();

    RefPtr gpuConnectionToWebProcess = factory->gpuConnectionToWebProcess();
    if (!gpuConnectionToWebProcess)
        return emptyString();

    return gpuConnectionToWebProcess->mediaKeysStorageDirectory();
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& RemoteLegacyCDMSessionProxy::logChannel() const
{
    return JOIN_LOG_CHANNEL_WITH_PREFIX(LOG_CHANNEL_PREFIX, EME);
}
#endif

std::optional<SharedPreferencesForWebProcess> RemoteLegacyCDMSessionProxy::sharedPreferencesForWebProcess() const
{
    if (!m_factory)
        return std::nullopt;

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG return m_factory->sharedPreferencesForWebProcess();
}

RefPtr<WebCore::LegacyCDMSession> RemoteLegacyCDMSessionProxy::protectedSession() const
{
    return m_session;
}

} // namespace WebKit

#endif
