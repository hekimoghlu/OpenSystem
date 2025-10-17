/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#include "StorageNamespaceProvider.h"

#include "Document.h"
#include "Page.h"
#include "SecurityOriginData.h"
#include "StorageArea.h"
#include "StorageNamespace.h"

namespace WebCore {

// Suggested by the HTML5 spec.
unsigned localStorageDatabaseQuotaInBytes = 5 * 1024 * 1024;

StorageNamespaceProvider::StorageNamespaceProvider()
{
}

StorageNamespaceProvider::~StorageNamespaceProvider()
{
}

Ref<StorageArea> StorageNamespaceProvider::localStorageArea(Document& document)
{
    // This StorageNamespaceProvider was retrieved from the Document's Page,
    // so the Document had better still actually have a Page.
    ASSERT(document.page());

    RefPtr<StorageNamespace> storageNamespace;
    if (document.canAccessResource(ScriptExecutionContext::ResourceType::LocalStorage) == ScriptExecutionContext::HasResourceAccess::DefaultForThirdParty)
        storageNamespace = &transientLocalStorageNamespace(document.topOrigin(), document.page()->sessionID());
    else
        storageNamespace = &localStorageNamespace(document.page()->sessionID());

    return storageNamespace->storageArea(document.securityOrigin());
}

Ref<StorageArea> StorageNamespaceProvider::sessionStorageArea(Document& document)
{
    // This StorageNamespaceProvider was retrieved from the Document's Page,
    // so the Document had better still actually have a Page.
    ASSERT(document.page());

    return sessionStorageNamespace(document.topOrigin(), *document.page())->storageArea(document.securityOrigin());
}

StorageNamespace& StorageNamespaceProvider::localStorageNamespace(PAL::SessionID sessionID)
{
    if (!m_localStorageNamespace)
        m_localStorageNamespace = createLocalStorageNamespace(localStorageDatabaseQuotaInBytes, sessionID);

    ASSERT(m_localStorageNamespace->sessionID() == sessionID);
    return *m_localStorageNamespace;
}

StorageNamespace& StorageNamespaceProvider::transientLocalStorageNamespace(SecurityOrigin& securityOrigin, PAL::SessionID sessionID)
{
    auto& slot = m_transientLocalStorageNamespaces.add(securityOrigin.data(), nullptr).iterator->value;
    if (!slot)
        slot = createTransientLocalStorageNamespace(securityOrigin, localStorageDatabaseQuotaInBytes, sessionID);

    ASSERT(slot->sessionID() == sessionID);
    return *slot;
}

void StorageNamespaceProvider::setSessionIDForTesting(PAL::SessionID newSessionID)
{
    if (m_localStorageNamespace && newSessionID != m_localStorageNamespace->sessionID())
        m_localStorageNamespace->setSessionIDForTesting(newSessionID);
    
    for (auto& transientLocalStorageNamespace : m_transientLocalStorageNamespaces.values()) {
        if (newSessionID != transientLocalStorageNamespace->sessionID())
            m_localStorageNamespace->setSessionIDForTesting(newSessionID);
    }
}

}
