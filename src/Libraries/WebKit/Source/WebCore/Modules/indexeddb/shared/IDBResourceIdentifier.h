/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 5, 2022.
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

#include "ProcessIdentifier.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/Hasher.h>

namespace WebCore {

class IDBRequest;

namespace IDBClient {
class IDBConnectionProxy;
}

namespace IDBServer {
class IDBConnectionToClient;
}

using IDBConnectionIdentifier = ProcessIdentifier;
struct IDBResourceIdentifierHashTraits;

class IDBResourceIdentifier {
public:
    explicit IDBResourceIdentifier(const IDBClient::IDBConnectionProxy&);
    IDBResourceIdentifier(const IDBClient::IDBConnectionProxy&, const IDBRequest&);
    explicit IDBResourceIdentifier(const IDBServer::IDBConnectionToClient&);

    bool isEmpty() const
    {
        return !m_resourceNumber && !m_idbConnectionIdentifier;
    }

    friend bool operator==(const IDBResourceIdentifier&, const IDBResourceIdentifier&) = default;
    
    std::optional<IDBConnectionIdentifier> connectionIdentifier() const { return m_idbConnectionIdentifier; }

    WEBCORE_EXPORT IDBResourceIdentifier isolatedCopy() const;

#if !LOG_DISABLED
    String loggingString() const;
#endif

    WEBCORE_EXPORT IDBResourceIdentifier();
private:
    friend struct IPC::ArgumentCoder<IDBResourceIdentifier, void>;
    friend struct IDBResourceIdentifierHashTraits;
    friend void add(Hasher&, const IDBResourceIdentifier&);

    WEBCORE_EXPORT IDBResourceIdentifier(std::optional<IDBConnectionIdentifier>, uint64_t resourceIdentifier);

    Markable<IDBConnectionIdentifier> m_idbConnectionIdentifier;
    uint64_t m_resourceNumber { 0 };
};

inline void add(Hasher& hasher, const IDBResourceIdentifier& identifier)
{
    add(hasher, identifier.m_idbConnectionIdentifier, identifier.m_resourceNumber);
}

struct IDBResourceIdentifierHash {
    static unsigned hash(const IDBResourceIdentifier& a) { return computeHash(a); }
    static bool equal(const IDBResourceIdentifier& a, const IDBResourceIdentifier& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

struct IDBResourceIdentifierHashTraits : WTF::CustomHashTraits<IDBResourceIdentifier> {
    static constexpr bool hasIsEmptyValueFunction = true;
    static constexpr bool emptyValueIsZero = false;
    static constexpr uint64_t resourceNumberDeletedValue = -1;

    static IDBResourceIdentifier emptyValue()
    {
        return { };
    }

    static bool isEmptyValue(const IDBResourceIdentifier& identifier)
    {
        return identifier.isEmpty();
    }

    static void constructDeletedValue(IDBResourceIdentifier& identifier)
    {
        identifier.m_resourceNumber = resourceNumberDeletedValue;
    }

    static bool isDeletedValue(const IDBResourceIdentifier& identifier)
    {
        return identifier.m_resourceNumber == resourceNumberDeletedValue;
    }
};

} // namespace WebCore

namespace WTF {

template<> struct HashTraits<WebCore::IDBResourceIdentifier> : WebCore::IDBResourceIdentifierHashTraits { };
template<> struct DefaultHash<WebCore::IDBResourceIdentifier> : WebCore::IDBResourceIdentifierHash { };

inline WebCore::IDBConnectionIdentifier crossThreadCopy(WebCore::IDBConnectionIdentifier identifier)
{
    return identifier;
}

} // namespace WTF
