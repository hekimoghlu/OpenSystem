/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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
#include "CDM.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "CDMFactory.h"
#include "CDMPrivate.h"
#include "ContextDestructionObserverInlines.h"
#include "Document.h"
#include "InitDataRegistry.h"
#include "MediaKeysRequirement.h"
#include "MediaPlayer.h"
#include "NotImplemented.h"
#include "Page.h"
#include "ParsedContentType.h"
#include "ScriptExecutionContext.h"
#include "SecurityOrigin.h"
#include "SecurityOriginData.h"
#include "Settings.h"
#include "SharedBuffer.h"
#include <wtf/FileSystem.h>
#include <wtf/Logger.h>
#include <wtf/LoggerHelper.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

bool CDM::supportsKeySystem(const String& keySystem)
{
    for (auto* factory : CDMFactory::registeredFactories()) {
        if (factory->supportsKeySystem(keySystem))
            return true;
    }
    return false;
}

Ref<CDM> CDM::create(Document& document, const String& keySystem)
{
    return adoptRef(*new CDM(document, keySystem));
}

CDM::CDM(Document& document, const String& keySystem)
    : ContextDestructionObserver(&document)
#if !RELEASE_LOG_DISABLED
    , m_logger(document.logger())
    , m_logIdentifier(LoggerHelper::uniqueLogIdentifier())
#endif
    , m_keySystem(keySystem)
{
    ASSERT(supportsKeySystem(keySystem));
    for (auto* factory : CDMFactory::registeredFactories()) {
        if (factory->supportsKeySystem(keySystem)) {
            m_private = factory->createCDM(keySystem, *this);
#if !RELEASE_LOG_DISABLED
            m_private->setLogIdentifier(m_logIdentifier);
#endif
            break;
        }
    }
}

CDM::~CDM() = default;

void CDM::getSupportedConfiguration(MediaKeySystemConfiguration&& candidateConfiguration, SupportedConfigurationCallback&& callback)
{
    // https://w3c.github.io/encrypted-media/#get-supported-configuration
    // W3C Editor's Draft 09 November 2016
    // Implemented in CDMPrivate::getSupportedConfiguration()

    RefPtr document = downcast<Document>(scriptExecutionContext());
    if (!document || !m_private) {
        callback(std::nullopt);
        return;
    }

    auto access = CDMPrivate::LocalStorageAccess::Allowed;
    bool isEphemeral = !document->page() || document->page()->sessionID().isEphemeral();
    if (isEphemeral || document->canAccessResource(ScriptExecutionContext::ResourceType::LocalStorage) == ScriptExecutionContext::HasResourceAccess::No)
        access = CDMPrivate::LocalStorageAccess::NotAllowed;
    m_private->getSupportedConfiguration(WTFMove(candidateConfiguration), access, WTFMove(callback));
}

void CDM::loadAndInitialize()
{
    if (m_private)
        m_private->loadAndInitialize();
}

RefPtr<CDMInstance> CDM::createInstance()
{
    if (!m_private)
        return nullptr;
    auto instance = m_private->createInstance();
    if (instance)
        instance->setStorageDirectory(storageDirectory());
    return instance;
}

bool CDM::supportsServerCertificates() const
{
    return m_private && m_private->supportsServerCertificates();
}

bool CDM::supportsSessions() const
{
    return m_private && m_private->supportsSessions();
}

bool CDM::supportsInitDataType(const AtomString& initDataType) const
{
    return m_private && m_private->supportedInitDataTypes().contains(initDataType);
}

RefPtr<SharedBuffer> CDM::sanitizeInitData(const AtomString& initDataType, const SharedBuffer& initData)
{
    return InitDataRegistry::shared().sanitizeInitData(initDataType, initData);
}

bool CDM::supportsInitData(const AtomString& initDataType, const SharedBuffer& initData)
{
    return m_private && m_private->supportsInitData(initDataType, initData);
}

RefPtr<SharedBuffer> CDM::sanitizeResponse(const SharedBuffer& response)
{
    if (!m_private)
        return nullptr;
    return m_private->sanitizeResponse(response);
}

std::optional<String> CDM::sanitizeSessionId(const String& sessionId)
{
    if (!m_private)
        return std::nullopt;
    return m_private->sanitizeSessionId(sessionId);
}

String CDM::storageDirectory() const
{
    RefPtr document = downcast<Document>(scriptExecutionContext());
    return document ? document->mediaKeysStorageDirectory() : emptyString();
}

}

#endif
