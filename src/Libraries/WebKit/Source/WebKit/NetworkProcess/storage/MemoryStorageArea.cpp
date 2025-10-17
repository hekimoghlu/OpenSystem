/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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
#include "MemoryStorageArea.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(MemoryStorageArea);

Ref<MemoryStorageArea> MemoryStorageArea::create(const WebCore::ClientOrigin& origin, StorageAreaBase::StorageType type)
{
    return adoptRef(*new MemoryStorageArea(origin, type));
}

MemoryStorageArea::MemoryStorageArea(const WebCore::ClientOrigin& origin, StorageAreaBase::StorageType type)
    : StorageAreaBase(WebCore::StorageMap::noQuota, origin)
    , m_map(WebCore::StorageMap(WebCore::StorageMap::noQuota))
    , m_storageType(type)
{
}

bool MemoryStorageArea::isEmpty()
{
    return !m_map.length();
}

void MemoryStorageArea::clear()
{
    m_map.clear();
    notifyListenersAboutClear();
}

HashMap<String, String> MemoryStorageArea::allItems()
{
    return m_map.items();
}

Expected<void, StorageError> MemoryStorageArea::setItem(std::optional<IPC::Connection::UniqueID> connection, std::optional<StorageAreaImplIdentifier> storageAreaImplID, String&& key, String&& value, const String& urlString)
{
    String oldValue;
    bool hasQuotaError = false;
    m_map.setItem(key, value, oldValue, hasQuotaError);
    if (hasQuotaError)
        return makeUnexpected(StorageError::QuotaExceeded);

    if (connection && storageAreaImplID)
        dispatchEvents(*connection, *storageAreaImplID, key, oldValue, value, urlString);

    return { };
}

Expected<void, StorageError> MemoryStorageArea::removeItem(IPC::Connection::UniqueID connection, StorageAreaImplIdentifier storageAreaImplID, const String& key, const String& urlString)
{
    String oldValue;
    m_map.removeItem(key, oldValue);
    dispatchEvents(connection, storageAreaImplID, key, oldValue, String(), urlString);

    return { };
}

Expected<void, StorageError> MemoryStorageArea::clear(IPC::Connection::UniqueID connection, StorageAreaImplIdentifier implIdentifier, const String& urlString)
{
    if (!m_map.length())
        return { };

    m_map.clear();
    dispatchEvents(connection, implIdentifier, String(), String(), String(), urlString);

    return { };
}

Ref<MemoryStorageArea> MemoryStorageArea::clone() const
{
    Ref storageArea = MemoryStorageArea::create(origin(), m_storageType);
    storageArea->m_map = m_map;
    return storageArea;
}

} // namespace WebKit
