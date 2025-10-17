/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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

#include "ClientOrigin.h"
#include "SecurityOriginData.h"
#include <wtf/ArgumentCoder.h>

namespace WebCore {

class SecurityOrigin;

class IDBDatabaseIdentifier {
public:
    IDBDatabaseIdentifier() { }
    IDBDatabaseIdentifier(WTF::HashTableDeletedValueType)
        : m_databaseName(WTF::HashTableDeletedValue)
    {
    }

    WEBCORE_EXPORT IDBDatabaseIdentifier(const String& databaseName, SecurityOriginData&& openingOrigin, SecurityOriginData&& mainFrameOrigin, bool isTransient = false);

    IDBDatabaseIdentifier isolatedCopy() const &;
    IDBDatabaseIdentifier isolatedCopy() &&;

    bool isHashTableDeletedValue() const
    {
        return m_databaseName.isHashTableDeletedValue();
    }

    bool isValid() const
    {
        return !m_databaseName.isNull()
            && !m_databaseName.isHashTableDeletedValue();
    }

    bool isEmpty() const
    {
        return m_databaseName.isNull();
    }

    friend bool operator==(const IDBDatabaseIdentifier&, const IDBDatabaseIdentifier&) = default;

    const String& databaseName() const { return m_databaseName; }
    const ClientOrigin& origin() const { return m_origin; }
    bool isTransient() const { return m_isTransient; }

    String databaseDirectoryRelativeToRoot(const String& rootDirectory, ASCIILiteral versionString = "v1"_s) const;
    WEBCORE_EXPORT static String databaseDirectoryRelativeToRoot(const ClientOrigin&, const String& rootDirectory, ASCIILiteral versionString);
    WEBCORE_EXPORT static String optionalDatabaseDirectoryRelativeToRoot(const ClientOrigin&, const String& rootDirectory, ASCIILiteral versionString);

#if !LOG_DISABLED
    String loggingString() const;
#endif

    bool isRelatedToOrigin(const SecurityOriginData& other) const { return m_origin.isRelated(other); }

private:
    friend struct IPC::ArgumentCoder<IDBDatabaseIdentifier, void>;

    String m_databaseName;
    ClientOrigin m_origin;
    bool m_isTransient { false };
};

inline void add(Hasher& hasher, const IDBDatabaseIdentifier& identifier)
{
    add(hasher, identifier.databaseName(), identifier.origin(), identifier.isTransient());
}

struct IDBDatabaseIdentifierHash {
    static unsigned hash(const IDBDatabaseIdentifier& a) { return computeHash(a); }
    static bool equal(const IDBDatabaseIdentifier& a, const IDBDatabaseIdentifier& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

struct IDBDatabaseIdentifierHashTraits : SimpleClassHashTraits<IDBDatabaseIdentifier> {
    static const bool hasIsEmptyValueFunction = true;
    static const bool emptyValueIsZero = false;
    static bool isEmptyValue(const IDBDatabaseIdentifier& info) { return info.isEmpty(); }
};

} // namespace WebCore

namespace WTF {

template<> struct HashTraits<WebCore::IDBDatabaseIdentifier> : WebCore::IDBDatabaseIdentifierHashTraits { };
template<> struct DefaultHash<WebCore::IDBDatabaseIdentifier> : WebCore::IDBDatabaseIdentifierHash { };

} // namespace WTF
