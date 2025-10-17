/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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

#include "IDBIndexIdentifier.h"
#include "IDBIndexInfo.h"
#include "IDBKeyPath.h"
#include "IDBObjectStoreIdentifier.h"
#include <wtf/HashMap.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class IDBObjectStoreInfo {
public:
    WEBCORE_EXPORT IDBObjectStoreInfo(IDBObjectStoreIdentifier, const String& name, std::optional<IDBKeyPath>&&, bool autoIncrement, HashMap<IDBIndexIdentifier, IDBIndexInfo>&& = { });

    IDBObjectStoreIdentifier identifier() const { return m_identifier; }
    const String& name() const { return m_name; }
    const std::optional<IDBKeyPath>& keyPath() const { return m_keyPath; }
    bool autoIncrement() const { return m_autoIncrement; }

    void rename(const String& newName) { m_name = newName; }

    WEBCORE_EXPORT IDBObjectStoreInfo isolatedCopy() const &;
    WEBCORE_EXPORT IDBObjectStoreInfo isolatedCopy() &&;

    IDBIndexInfo createNewIndex(IDBIndexIdentifier, const String& name, IDBKeyPath&&, bool unique, bool multiEntry);
    void addExistingIndex(const IDBIndexInfo&);
    bool hasIndex(const String& name) const;
    bool hasIndex(IDBIndexIdentifier) const;
    IDBIndexInfo* infoForExistingIndex(const String& name);
    IDBIndexInfo* infoForExistingIndex(IDBIndexIdentifier);

    Vector<String> indexNames() const;
    const HashMap<IDBIndexIdentifier, IDBIndexInfo>& indexMap() const { return m_indexMap; }

    void deleteIndex(const String& indexName);
    void deleteIndex(IDBIndexIdentifier);

#if !LOG_DISABLED
    String loggingString(int indent = 0) const;
    String condensedLoggingString() const;
#endif

private:
    IDBObjectStoreIdentifier m_identifier;
    String m_name;
    std::optional<IDBKeyPath> m_keyPath;
    bool m_autoIncrement { false };

    HashMap<IDBIndexIdentifier, IDBIndexInfo> m_indexMap;
};

} // namespace WebCore

namespace WTF {

template<> struct HashTraits<WebCore::IDBObjectStoreInfo> : GenericHashTraits<WebCore::IDBObjectStoreInfo> {
    static constexpr bool emptyValueIsZero = false;
    static WebCore::IDBObjectStoreInfo emptyValue()
    {
        return WebCore::IDBObjectStoreInfo { HashTraits<WebCore::IDBObjectStoreIdentifier>::emptyValue(), { }, { }, false };
    }
    static bool isEmptyValue(const WebCore::IDBObjectStoreInfo& value) { return value.identifier().isHashTableEmptyValue(); }
};

} // namespace WTF
