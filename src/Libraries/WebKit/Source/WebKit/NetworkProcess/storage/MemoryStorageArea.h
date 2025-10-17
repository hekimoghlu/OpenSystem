/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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

#include "StorageAreaBase.h"
#include <WebCore/StorageMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class StorageMap;
}

namespace WebKit {

class MemoryStorageArea final : public StorageAreaBase, public RefCounted<MemoryStorageArea> {
    WTF_MAKE_TZONE_ALLOCATED(MemoryStorageArea);
public:
    static Ref<MemoryStorageArea> create(const WebCore::ClientOrigin&, StorageAreaBase::StorageType = StorageAreaBase::StorageType::Session);

    StorageAreaBase::Type type() const final { return StorageAreaBase::Type::Memory; }
    bool isEmpty() final;
    void clear() final;
    Ref<MemoryStorageArea> clone() const;

    // StorageAreaBase
    HashMap<String, String> allItems() final;
    Expected<void, StorageError> setItem(std::optional<IPC::Connection::UniqueID>, std::optional<StorageAreaImplIdentifier>, String&& key, String&& value, const String& urlString) final;
    
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    explicit MemoryStorageArea(const WebCore::ClientOrigin&, StorageAreaBase::StorageType);

    // StorageAreaBase
    StorageAreaBase::StorageType storageType() const final { return m_storageType; }
    Expected<void, StorageError> removeItem(IPC::Connection::UniqueID, StorageAreaImplIdentifier, const String& key, const String& urlString) final;
    Expected<void, StorageError> clear(IPC::Connection::UniqueID, StorageAreaImplIdentifier, const String& urlString) final;

    mutable WebCore::StorageMap m_map;
    StorageAreaBase::StorageType m_storageType;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::MemoryStorageArea)
    static bool isType(const WebKit::StorageAreaBase& area) { return area.type() == WebKit::StorageAreaBase::Type::Memory; }
SPECIALIZE_TYPE_TRAITS_END()
