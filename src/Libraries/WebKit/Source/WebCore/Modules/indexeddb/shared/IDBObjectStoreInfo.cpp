/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#include "IDBObjectStoreInfo.h"

#include <wtf/CrossThreadCopier.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

IDBObjectStoreInfo::IDBObjectStoreInfo(IDBObjectStoreIdentifier identifier, const String& name, std::optional<IDBKeyPath>&& keyPath, bool autoIncrement, HashMap<IDBIndexIdentifier, IDBIndexInfo>&& indexMap)
    : m_identifier(identifier)
    , m_name(name)
    , m_keyPath(WTFMove(keyPath))
    , m_autoIncrement(autoIncrement)
    , m_indexMap(WTFMove(indexMap))
{
}

IDBIndexInfo IDBObjectStoreInfo::createNewIndex(IDBIndexIdentifier indexID, const String& name, IDBKeyPath&& keyPath, bool unique, bool multiEntry)
{
    IDBIndexInfo info(indexID, m_identifier, name, WTFMove(keyPath), unique, multiEntry);
    m_indexMap.set(info.identifier(), info);
    return info;
}

void IDBObjectStoreInfo::addExistingIndex(const IDBIndexInfo& info)
{
    if (m_indexMap.contains(info.identifier()))
        LOG_ERROR("Adding an index '%s' with existing Index ID", info.name().utf8().data());

    m_indexMap.set(info.identifier(), info);
}

bool IDBObjectStoreInfo::hasIndex(const String& name) const
{
    for (auto& index : m_indexMap.values()) {
        if (index.name() == name)
            return true;
    }

    return false;
}

bool IDBObjectStoreInfo::hasIndex(IDBIndexIdentifier indexIdentifier) const
{
    return m_indexMap.contains(indexIdentifier);
}

IDBIndexInfo* IDBObjectStoreInfo::infoForExistingIndex(const String& name)
{
    for (auto& index : m_indexMap.values()) {
        if (index.name() == name)
            return &index;
    }

    return nullptr;
}

IDBIndexInfo* IDBObjectStoreInfo::infoForExistingIndex(IDBIndexIdentifier identifier)
{
    auto iterator = m_indexMap.find(identifier);
    if (iterator == m_indexMap.end())
        return nullptr;

    return &iterator->value;
}

IDBObjectStoreInfo IDBObjectStoreInfo::isolatedCopy() const &
{
    IDBObjectStoreInfo result = { m_identifier, m_name.isolatedCopy(), crossThreadCopy(m_keyPath), m_autoIncrement };
    result.m_indexMap = crossThreadCopy(m_indexMap);
    return result;
}

IDBObjectStoreInfo IDBObjectStoreInfo::isolatedCopy() &&
{
    IDBObjectStoreInfo result = { m_identifier, WTFMove(m_name).isolatedCopy(), crossThreadCopy(WTFMove(m_keyPath)), m_autoIncrement };
    result.m_indexMap = crossThreadCopy(WTFMove(m_indexMap));
    return result;
}

Vector<String> IDBObjectStoreInfo::indexNames() const
{
    return WTF::map(m_indexMap, [](auto& pair) -> String {
        return pair.value.name();
    });
}

void IDBObjectStoreInfo::deleteIndex(const String& indexName)
{
    auto* info = infoForExistingIndex(indexName);
    if (!info)
        return;

    m_indexMap.remove(info->identifier());
}

void IDBObjectStoreInfo::deleteIndex(IDBIndexIdentifier indexIdentifier)
{
    m_indexMap.remove(indexIdentifier);
}

#if !LOG_DISABLED

String IDBObjectStoreInfo::loggingString(int indent) const
{
    StringBuilder builder;
    for (int i = 0; i < indent; ++i)
        builder.append(' ');
    builder.append("Object store: "_s, m_name, m_identifier);
    for (auto index : m_indexMap.values())
        builder.append(index.loggingString(indent + 1), '\n');
    return builder.toString();
}

String IDBObjectStoreInfo::condensedLoggingString() const
{
    return makeString("<OS: "_s, m_name, " ("_s, m_identifier, ")>"_s);
}

#endif

} // namespace WebCore
