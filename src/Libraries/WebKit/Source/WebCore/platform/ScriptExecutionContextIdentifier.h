/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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
#include <wtf/Forward.h>
#include <wtf/UUID.h>

namespace WebCore {

template <>
class ProcessQualified<WTF::UUID> {
public:
    static ProcessQualified generate() { return { WTF::UUID::createVersion4Weak(), Process::identifier() }; }

    ProcessQualified(WTF::UUID object, ProcessIdentifier processIdentifier)
        : m_object(WTFMove(object))
        , m_processIdentifier(processIdentifier)
    {
    }

    ProcessQualified(WTF::HashTableDeletedValueType)
        : m_object(WTF::HashTableDeletedValue)
        , m_processIdentifier(WTF::HashTableDeletedValue)
    {
    }

    operator bool() const { return !!m_object; }

    const WTF::UUID& object() const { return m_object; }
    ProcessIdentifier processIdentifier() const { return m_processIdentifier; }

    bool isHashTableDeletedValue() const { return m_processIdentifier.isHashTableDeletedValue(); }

    friend bool operator==(const ProcessQualified&, const ProcessQualified&) = default;

    String toString() const { return m_object.toString(); }

    template<typename Encoder> void encode(Encoder& encoder) const { encoder << m_object << m_processIdentifier; }
    template<typename Decoder> static std::optional<ProcessQualified> decode(Decoder&);

    struct MarkableTraits {
        static bool isEmptyValue(const ProcessQualified<WTF::UUID>& identifier) { return WTF::UUID::MarkableTraits::isEmptyValue(identifier.object()); }
        static ProcessQualified<WTF::UUID> emptyValue() { return { WTF::UUID::MarkableTraits::emptyValue(), ProcessIdentifier::MarkableTraits::emptyValue() }; }
    };

private:
    WTF::UUID m_object;
    ProcessIdentifier m_processIdentifier;
};

inline void add(Hasher& hasher, const ProcessQualified<WTF::UUID>& uuid)
{
    // Since UUIDs are unique on their own, optimize by not hashing the process identifier.
    add(hasher, uuid.object());
}

template<typename Decoder> std::optional<ProcessQualified<WTF::UUID>> ProcessQualified<WTF::UUID>::decode(Decoder& decoder)
{
    std::optional<WTF::UUID> object;
    decoder >> object;
    if (!object)
        return std::nullopt;
    std::optional<ProcessIdentifier> processIdentifier;
    decoder >> processIdentifier;
    if (!processIdentifier)
        return std::nullopt;
    return { { *object, *processIdentifier } };
}

template <>
inline TextStream& operator<<(TextStream& ts, const ProcessQualified<WTF::UUID>& processQualified)
{
    ts << "ProcessQualified(" << processQualified.processIdentifier().toUInt64() << '-' << processQualified.object().toString() << ')';
    return ts;
}

using ScriptExecutionContextIdentifier = ProcessQualified<WTF::UUID>;

}
