/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "StorageAreaBase.h"

#include "StorageAreaMapMessages.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(StorageAreaBase);

uint64_t StorageAreaBase::nextMessageIdentifier()
{
    static std::atomic<uint64_t> currentIdentifier;
    return ++currentIdentifier;
}

StorageAreaBase::StorageAreaBase(unsigned quota, const WebCore::ClientOrigin& origin)
    : m_quota(quota)
    , m_origin(origin)
{
}

StorageAreaBase::~StorageAreaBase() = default;

void StorageAreaBase::addListener(IPC::Connection::UniqueID connection, StorageAreaMapIdentifier identifier)
{
    ASSERT(!m_listeners.contains(connection) || m_listeners.get(connection) == identifier);

    m_listeners.add(connection, identifier);
}

void StorageAreaBase::removeListener(IPC::Connection::UniqueID connection)
{
    m_listeners.remove(connection);
}

void StorageAreaBase::notifyListenersAboutClear()
{
    for (auto& [connection, identifier] : m_listeners)
        IPC::Connection::send(connection, Messages::StorageAreaMap::ClearCache(StorageAreaBase::nextMessageIdentifier()), identifier.toUInt64());
}

void StorageAreaBase::dispatchEvents(IPC::Connection::UniqueID sourceConnection, StorageAreaImplIdentifier sourceImplIdentifier, const String& key, const String& oldValue, const String& newValue, const String& urlString) const
{
    for (auto& [connection, identifier] : m_listeners) {
        std::optional<StorageAreaImplIdentifier> implIdentifier;
        if (connection == sourceConnection)
            implIdentifier = sourceImplIdentifier;
        IPC::Connection::send(connection, Messages::StorageAreaMap::DispatchStorageEvent(implIdentifier, key, oldValue, newValue, urlString, StorageAreaBase::nextMessageIdentifier()), identifier.toUInt64());
    }
}

} // namespace WebKit
