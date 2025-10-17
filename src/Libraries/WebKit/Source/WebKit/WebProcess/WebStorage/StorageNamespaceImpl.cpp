/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#include "StorageNamespaceImpl.h"

#include "NetworkProcessConnection.h"
#include "NetworkStorageManagerMessages.h"
#include "StorageAreaImpl.h"
#include "StorageAreaMap.h"
#include "WebPage.h"
#include "WebPageInlines.h"
#include "WebProcess.h"
#include <WebCore/LocalFrame.h>
#include <WebCore/SecurityOrigin.h>
#include <WebCore/Settings.h>
#include <WebCore/StorageType.h>

namespace WebKit {
using namespace WebCore;

Ref<StorageNamespaceImpl> StorageNamespaceImpl::createSessionStorageNamespace(Identifier identifier, PageIdentifier pageID, const WebCore::SecurityOrigin& topLevelOrigin, unsigned quotaInBytes)
{
    return adoptRef(*new StorageNamespaceImpl(StorageType::Session, pageID, &topLevelOrigin, quotaInBytes, identifier));
}

Ref<StorageNamespaceImpl> StorageNamespaceImpl::createLocalStorageNamespace(unsigned quotaInBytes)
{
    return adoptRef(*new StorageNamespaceImpl(StorageType::Local, std::nullopt, nullptr, quotaInBytes));
}

Ref<StorageNamespaceImpl> StorageNamespaceImpl::createTransientLocalStorageNamespace(WebCore::SecurityOrigin& topLevelOrigin, uint64_t quotaInBytes)
{
    return adoptRef(*new StorageNamespaceImpl(StorageType::TransientLocal, std::nullopt, &topLevelOrigin, quotaInBytes));
}

StorageNamespaceImpl::StorageNamespaceImpl(WebCore::StorageType storageType, const std::optional<PageIdentifier>& pageIdentifier, const WebCore::SecurityOrigin* topLevelOrigin, unsigned quotaInBytes, std::optional<Identifier> storageNamespaceID)
    : m_storageType(storageType)
    , m_sessionPageID(pageIdentifier)
    , m_topLevelOrigin(topLevelOrigin)
    , m_quotaInBytes(quotaInBytes)
    , m_storageNamespaceID(storageNamespaceID)
{
    ASSERT(storageType == StorageType::Session || !m_sessionPageID);
}

StorageNamespaceImpl::~StorageNamespaceImpl() = default;

PAL::SessionID StorageNamespaceImpl::sessionID() const
{
    return WebProcess::singleton().sessionID();
}

void StorageNamespaceImpl::destroyStorageAreaMap(StorageAreaMap& map)
{
    m_storageAreaMaps.remove(map.securityOrigin().data());
}

Ref<StorageArea> StorageNamespaceImpl::storageArea(const SecurityOrigin& securityOrigin)
{
    auto& map = m_storageAreaMaps.ensure(securityOrigin.data(), [&] {
        return StorageAreaMap::create(*this, securityOrigin);
    }).iterator->value;
    return StorageAreaImpl::create(map);
}

Ref<StorageNamespace> StorageNamespaceImpl::copy(Page& newPage)
{
    ASSERT(m_storageNamespaceID);
    ASSERT(m_storageType == StorageType::Session);

    RefPtr webPage = WebPage::fromCorePage(newPage);
    return adoptRef(*new StorageNamespaceImpl(m_storageType, webPage->identifier(), m_topLevelOrigin.get(), m_quotaInBytes, webPage->sessionStorageNamespaceIdentifier()));
}

void StorageNamespaceImpl::setSessionIDForTesting(PAL::SessionID)
{
    ASSERT_NOT_REACHED();
}

} // namespace WebKit
