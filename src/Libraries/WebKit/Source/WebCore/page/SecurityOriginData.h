/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#include "ProcessQualified.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/Hasher.h>
#include <wtf/Markable.h>
#include <wtf/URL.h>

namespace WebCore {

class LocalFrame;
class SecurityOrigin;

enum class OpaqueOriginIdentifierType { };
using OpaqueOriginIdentifier = AtomicObjectIdentifier<OpaqueOriginIdentifierType>;

class SecurityOriginData {
public:
    struct Tuple {
        String protocol;
        String host;
        std::optional<uint16_t> port;

        friend bool operator==(const Tuple&, const Tuple&) = default;
        Tuple isolatedCopy() const & { return { protocol.isolatedCopy(), host.isolatedCopy(), port }; }
        Tuple isolatedCopy() && { return { WTFMove(protocol).isolatedCopy(), WTFMove(host).isolatedCopy(), port }; }
    };

    SecurityOriginData() = default;
    SecurityOriginData(const String& protocol, const String& host, std::optional<uint16_t> port)
        : m_data { Tuple { protocol, host, port } }
    {
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(!isHashTableDeletedValue());
    }
    explicit SecurityOriginData(ProcessQualified<OpaqueOriginIdentifier> opaqueOriginIdentifier)
        : m_data(opaqueOriginIdentifier) { }
    explicit SecurityOriginData(std::variant<Tuple, ProcessQualified<OpaqueOriginIdentifier>>&& data)
        : m_data(WTFMove(data)) { }
    SecurityOriginData(WTF::HashTableDeletedValueType)
        : m_data { Tuple { WTF::HashTableDeletedValue, { }, { } } } { }
    
    WEBCORE_EXPORT static SecurityOriginData fromFrame(LocalFrame*);
    WEBCORE_EXPORT static SecurityOriginData fromURL(const URL&);
    WEBCORE_EXPORT static SecurityOriginData fromURLWithoutStrictOpaqueness(const URL&);

    static SecurityOriginData createOpaque()
    {
        return SecurityOriginData { ProcessQualified<OpaqueOriginIdentifier>::generate() };
    }

    WEBCORE_EXPORT Ref<SecurityOrigin> securityOrigin() const;

    // FIXME <rdar://9018386>: We should be sending more state across the wire than just the protocol,
    // host, and port.

    const String& protocol() const
    {
        return switchOn(m_data, [] (const Tuple& tuple) -> const String& {
            return tuple.protocol;
        }, [] (const ProcessQualified<OpaqueOriginIdentifier>&) -> const String& {
            return emptyString();
        });
    }
    const String& host() const
    {
        return switchOn(m_data, [] (const Tuple& tuple) -> const String&  {
            return tuple.host;
        }, [] (const ProcessQualified<OpaqueOriginIdentifier>&) -> const String& {
            return emptyString();
        });
    }
    std::optional<uint16_t> port() const
    {
        return switchOn(m_data, [] (const Tuple& tuple) -> std::optional<uint16_t> {
            return tuple.port;
        }, [] (const ProcessQualified<OpaqueOriginIdentifier>&) -> std::optional<uint16_t> {
            return std::nullopt;
        });
    }
    void setPort(std::optional<uint16_t> port)
    {
        switchOn(m_data, [port] (Tuple& tuple) {
            tuple.port = port;
        }, [] (const ProcessQualified<OpaqueOriginIdentifier>&) {
            ASSERT_NOT_REACHED();
        });
    }
    std::optional<ProcessQualified<OpaqueOriginIdentifier>> opaqueOriginIdentifier() const
    {
        return switchOn(m_data, [] (const Tuple&) {
            return std::optional<ProcessQualified<OpaqueOriginIdentifier>> { };
        }, [] (const ProcessQualified<OpaqueOriginIdentifier>& identifier) -> std::optional<ProcessQualified<OpaqueOriginIdentifier>> {
            return identifier;
        });
    }
    
    WEBCORE_EXPORT SecurityOriginData isolatedCopy() const &;
    WEBCORE_EXPORT SecurityOriginData isolatedCopy() &&;

    // Serialize the security origin to a string that could be used as part of
    // file names. This format should be used in storage APIs only.
    WEBCORE_EXPORT String databaseIdentifier() const;
    WEBCORE_EXPORT String optionalDatabaseIdentifier() const;
    WEBCORE_EXPORT static std::optional<SecurityOriginData> fromDatabaseIdentifier(StringView);

    bool isNull() const
    {
        return switchOn(m_data, [] (const Tuple& tuple) {
            return tuple.protocol.isNull() && tuple.host.isNull() && tuple.port == std::nullopt;
        }, [] (const ProcessQualified<OpaqueOriginIdentifier>&) {
            return false;
        });
    }
    bool isOpaque() const
    {
        return std::holds_alternative<ProcessQualified<OpaqueOriginIdentifier>>(m_data);
    }

    bool isHashTableDeletedValue() const
    {
        return switchOn(m_data, [] (const Tuple& tuple) {
            return tuple.protocol.isHashTableDeletedValue();
        }, [] (const ProcessQualified<OpaqueOriginIdentifier>&) {
            return false;
        });
    }
    
    WEBCORE_EXPORT String toString() const;

    WEBCORE_EXPORT URL toURL() const;

#if !LOG_DISABLED
    String debugString() const { return toString(); }
#endif

    static bool shouldTreatAsOpaqueOrigin(const URL&);
    
    const std::variant<Tuple, ProcessQualified<OpaqueOriginIdentifier>>& data() const { return m_data; }
private:
    std::variant<Tuple, ProcessQualified<OpaqueOriginIdentifier>> m_data;
};

WEBCORE_EXPORT bool operator==(const SecurityOriginData&, const SecurityOriginData&);

inline void add(Hasher& hasher, const SecurityOriginData& data)
{
    add(hasher, data.data());
}

inline void add(Hasher& hasher, const SecurityOriginData::Tuple& tuple)
{
    add(hasher, tuple.protocol, tuple.host, tuple.port);
}

struct SecurityOriginDataHashTraits : SimpleClassHashTraits<SecurityOriginData> {
    static const bool hasIsEmptyValueFunction = true;
    static const bool emptyValueIsZero = false;
    static bool isEmptyValue(const SecurityOriginData& data) { return data.isNull(); }
};

struct SecurityOriginDataHash {
    static unsigned hash(const SecurityOriginData& data) { return computeHash(data); }
    static unsigned hash(const std::optional<SecurityOriginData>& data) { return computeHash(data); }
    static bool equal(const SecurityOriginData& a, const SecurityOriginData& b) { return a == b; }
    static bool equal(const std::optional<SecurityOriginData>& a, const std::optional<SecurityOriginData>& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

struct SecurityOriginDataMarkableTraits {
    static bool isEmptyValue(const SecurityOriginData& value) { return value.isNull(); }
    static SecurityOriginData emptyValue() { return { }; }
};
} // namespace WebCore

namespace WTF {

template<> struct HashTraits<WebCore::SecurityOriginData> : WebCore::SecurityOriginDataHashTraits { };
template<> struct DefaultHash<WebCore::SecurityOriginData> : WebCore::SecurityOriginDataHash { };
template<> struct DefaultHash<std::optional<WebCore::SecurityOriginData>> : WebCore::SecurityOriginDataHash { };

} // namespace WTF
