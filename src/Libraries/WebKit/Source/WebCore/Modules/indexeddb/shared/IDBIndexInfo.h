/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 24, 2024.
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
#include "IDBKeyPath.h"
#include "IDBObjectStoreIdentifier.h"
#include <wtf/HashTraits.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class IDBIndexInfo {
public:
    WEBCORE_EXPORT IDBIndexInfo(IDBIndexIdentifier, IDBObjectStoreIdentifier, const String& name, IDBKeyPath&&, bool unique, bool multiEntry);

    WEBCORE_EXPORT IDBIndexInfo isolatedCopy() const &;
    WEBCORE_EXPORT IDBIndexInfo isolatedCopy() &&;

    IDBIndexIdentifier identifier() const { return m_identifier; }
    IDBObjectStoreIdentifier objectStoreIdentifier() const { return m_objectStoreIdentifier; }
    const String& name() const { return m_name; }
    const IDBKeyPath& keyPath() const { return m_keyPath; }
    bool unique() const { return m_unique; }
    bool multiEntry() const { return m_multiEntry; }

    void rename(const String& newName) { m_name = newName; }

#if !LOG_DISABLED
    String loggingString(int indent = 0) const;
    String condensedLoggingString() const;
#endif

    void setIdentifier(IDBIndexIdentifier identifier) { m_identifier = identifier; }
private:
    IDBIndexIdentifier m_identifier;
    IDBObjectStoreIdentifier m_objectStoreIdentifier;
    String m_name;
    IDBKeyPath m_keyPath;
    bool m_unique { true };
    bool m_multiEntry { false };
};

} // namespace WebCore

namespace WTF {

template<> struct HashTraits<WebCore::IDBIndexInfo> : GenericHashTraits<WebCore::IDBIndexInfo> {
    static constexpr bool emptyValueIsZero = false;
    static WebCore::IDBIndexInfo emptyValue()
    {
        return WebCore::IDBIndexInfo { HashTraits<WebCore::IDBIndexIdentifier>::emptyValue(), HashTraits<WebCore::IDBObjectStoreIdentifier>::emptyValue(), { }, { }, false, false };
    }
    static bool isEmptyValue(const WebCore::IDBIndexInfo& value) { return value.objectStoreIdentifier().isHashTableEmptyValue(); }
};

} // namespace WTF
