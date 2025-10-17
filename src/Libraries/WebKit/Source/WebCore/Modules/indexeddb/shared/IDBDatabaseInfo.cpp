/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
#include "IDBDatabaseInfo.h"

#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(IDBDatabaseInfo);

IDBDatabaseInfo::IDBDatabaseInfo(const String& name, uint64_t version, uint64_t maxIndexID, HashMap<IDBObjectStoreIdentifier, IDBObjectStoreInfo>&& objectStoreMap)
    : m_name(name)
    , m_version(version)
    , m_maxIndexID(maxIndexID)
    , m_objectStoreMap(WTFMove(objectStoreMap))
{
}

IDBDatabaseInfo::IDBDatabaseInfo(const IDBDatabaseInfo& other, IsolatedCopyTag)
    : m_name(other.m_name.isolatedCopy())
    , m_version(other.m_version)
    , m_maxIndexID(other.m_maxIndexID)
{
    for (const auto& entry : other.m_objectStoreMap)
        m_objectStoreMap.set(entry.key, entry.value.isolatedCopy());
}

IDBDatabaseInfo IDBDatabaseInfo::isolatedCopy() const
{
    return { *this, IDBDatabaseInfo::IsolatedCopy };
}

bool IDBDatabaseInfo::hasObjectStore(const String& name) const
{
    for (auto& objectStore : m_objectStoreMap.values()) {
        if (objectStore.name() == name)
            return true;
    }

    return false;
}

IDBObjectStoreInfo IDBDatabaseInfo::createNewObjectStore(const String& name, std::optional<IDBKeyPath>&& keyPath, bool autoIncrement)
{
    // ObjectIdentifier generation begins anew from 1 each time the program is restarted. But the IDBObjectStores
    // held in m_objectStoreMap and their identifiers are perisisted to disk. So we must ensure that newly created
    // identifiers do not conflict with identifiers for pre-existing IDBObjectStores loaded from disk.
    auto objectStoreIdentifier = IDBObjectStoreIdentifier::generate();
    while (m_objectStoreMap.contains(objectStoreIdentifier))
        objectStoreIdentifier = IDBObjectStoreIdentifier::generate();

    IDBObjectStoreInfo info(objectStoreIdentifier, name, WTFMove(keyPath), autoIncrement);
    m_objectStoreMap.set(info.identifier(), info);
    return info;
}

void IDBDatabaseInfo::addExistingObjectStore(const IDBObjectStoreInfo& info)
{
    ASSERT(!m_objectStoreMap.contains(info.identifier()));
    m_objectStoreMap.set(info.identifier(), info);
}

IDBObjectStoreInfo* IDBDatabaseInfo::getInfoForExistingObjectStore(IDBObjectStoreIdentifier objectStoreIdentifier)
{
    auto iterator = m_objectStoreMap.find(objectStoreIdentifier);
    if (iterator == m_objectStoreMap.end())
        return nullptr;

    return &iterator->value;
}

IDBObjectStoreInfo* IDBDatabaseInfo::getInfoForExistingObjectStore(const String& name)
{
    for (auto& objectStore : m_objectStoreMap.values()) {
        if (objectStore.name() == name)
            return &objectStore;
    }

    return nullptr;
}

const IDBObjectStoreInfo* IDBDatabaseInfo::infoForExistingObjectStore(IDBObjectStoreIdentifier objectStoreIdentifier) const
{
    return const_cast<IDBDatabaseInfo*>(this)->getInfoForExistingObjectStore(objectStoreIdentifier);
}

IDBObjectStoreInfo* IDBDatabaseInfo::infoForExistingObjectStore(IDBObjectStoreIdentifier objectStoreIdentifier)
{
    return getInfoForExistingObjectStore(objectStoreIdentifier);
}

const IDBObjectStoreInfo* IDBDatabaseInfo::infoForExistingObjectStore(const String& name) const
{
    return const_cast<IDBDatabaseInfo*>(this)->getInfoForExistingObjectStore(name);
}

IDBObjectStoreInfo* IDBDatabaseInfo::infoForExistingObjectStore(const String& name)
{
    return getInfoForExistingObjectStore(name);
}

void IDBDatabaseInfo::renameObjectStore(IDBObjectStoreIdentifier objectStoreIdentifier, const String& newName)
{
    auto* info = infoForExistingObjectStore(objectStoreIdentifier);
    if (!info)
        return;

    info->rename(newName);
}

Vector<String> IDBDatabaseInfo::objectStoreNames() const
{
    return WTF::map(m_objectStoreMap, [](auto& pair) {
        return pair.value.name();
    });
}

void IDBDatabaseInfo::deleteObjectStore(const String& objectStoreName)
{
    auto* info = infoForExistingObjectStore(objectStoreName);
    if (!info)
        return;

    m_objectStoreMap.remove(info->identifier());
}

void IDBDatabaseInfo::deleteObjectStore(IDBObjectStoreIdentifier objectStoreIdentifier)
{
    m_objectStoreMap.remove(objectStoreIdentifier);
}

#if !LOG_DISABLED

String IDBDatabaseInfo::loggingString() const
{
    StringBuilder builder;
    builder.append("Database:"_s, m_name, " version "_s, m_version, '\n');
    for (auto& objectStore : m_objectStoreMap.values())
        builder.append(objectStore.loggingString(1), '\n');
    return builder.toString();
}

#endif

void IDBDatabaseInfo::setMaxIndexID(uint64_t maxIndexID)
{
    ASSERT(maxIndexID > m_maxIndexID || (!maxIndexID && !m_maxIndexID));
    m_maxIndexID = maxIndexID;
}

} // namespace WebCore
