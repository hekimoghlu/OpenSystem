/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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

#include "IdentifierTypes.h"
#include "StorageNamespaceIdentifier.h"
#include <WebCore/PageIdentifier.h>
#include <WebCore/SecurityOriginData.h>
#include <WebCore/SecurityOriginHash.h>
#include <WebCore/StorageArea.h>
#include <WebCore/StorageMap.h>
#include <WebCore/StorageNamespace.h>
#include <WebCore/StorageType.h>
#include <pal/SessionID.h>
#include <wtf/HashMap.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class StorageAreaMap;
class WebPage;

class StorageNamespaceImpl final : public WebCore::StorageNamespace, public CanMakeWeakPtr<StorageNamespaceImpl> {
public:
    using Identifier = StorageNamespaceIdentifier;

    static Ref<StorageNamespaceImpl> createSessionStorageNamespace(Identifier, WebCore::PageIdentifier, const WebCore::SecurityOrigin&, unsigned quotaInBytes);
    static Ref<StorageNamespaceImpl> createLocalStorageNamespace(unsigned quotaInBytes);
    static Ref<StorageNamespaceImpl> createTransientLocalStorageNamespace(WebCore::SecurityOrigin& topLevelOrigin, uint64_t quotaInBytes);

    virtual ~StorageNamespaceImpl();

    WebCore::StorageType storageType() const { return m_storageType; }
    std::optional<Identifier> storageNamespaceID() const { return m_storageNamespaceID; }
    WebCore::PageIdentifier sessionStoragePageID() const;
    const WebCore::SecurityOrigin* topLevelOrigin() const final { return m_topLevelOrigin.get(); }
    unsigned quotaInBytes() const { return m_quotaInBytes; }
    PAL::SessionID sessionID() const override;

    void destroyStorageAreaMap(StorageAreaMap&);

    void setSessionIDForTesting(PAL::SessionID) override;

private:
    StorageNamespaceImpl(WebCore::StorageType, const std::optional<WebCore::PageIdentifier>&, const WebCore::SecurityOrigin* topLevelOrigin, unsigned quotaInBytes, std::optional<Identifier> = std::nullopt);

    Ref<WebCore::StorageArea> storageArea(const WebCore::SecurityOrigin&) final;
    uint64_t storageAreaMapCountForTesting() const final { return m_storageAreaMaps.size(); }

    // FIXME: This is only valid for session storage and should probably be moved to a subclass.
    Ref<WebCore::StorageNamespace> copy(WebCore::Page&) override;

    const WebCore::StorageType m_storageType;
    std::optional<WebCore::PageIdentifier> m_sessionPageID;

    // Used for transient local storage and session storage namespaces, nullptr otherwise.
    const RefPtr<const WebCore::SecurityOrigin> m_topLevelOrigin;
    const unsigned m_quotaInBytes;
    std::optional<Identifier> m_storageNamespaceID;

    HashMap<WebCore::SecurityOriginData, Ref<StorageAreaMap>> m_storageAreaMaps;
};

inline WebCore::PageIdentifier StorageNamespaceImpl::sessionStoragePageID() const
{
    ASSERT(m_storageType == WebCore::StorageType::Session);
    return *m_sessionPageID;
}

} // namespace WebKit
