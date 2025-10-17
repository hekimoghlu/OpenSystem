/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include "PasteboardCustomData.h"

#include "SharedBuffer.h"
#include <wtf/URLParser.h>
#include <wtf/persistence/PersistentCoders.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

static std::variant<String, Ref<SharedBuffer>> copyPlatformData(const std::variant<String, Ref<SharedBuffer>>& other)
{
    if (std::holds_alternative<String>(other))
        return { std::get<String>(other) };

    if (std::holds_alternative<Ref<SharedBuffer>>(other))
        return { std::get<Ref<SharedBuffer>>(other).copyRef() };

    return { };
}

PasteboardCustomData::Entry::Entry(const Entry& entry)
    : type(entry.type)
    , customData(entry.customData)
    , platformData(copyPlatformData(entry.platformData))
{
}

PasteboardCustomData::Entry::Entry(const String& dataType)
    : type(dataType)
{
}

PasteboardCustomData::Entry::Entry() = default;
PasteboardCustomData::Entry::Entry(Entry&&) = default;

PasteboardCustomData::Entry::Entry(const String& type, const String& customData, const std::variant<String, Ref<WebCore::SharedBuffer>>& platformData)
    : type(type)
    , customData(customData)
    , platformData(platformData)
{
}

PasteboardCustomData::Entry& PasteboardCustomData::Entry::operator=(const Entry& entry)
{
    type = entry.type;
    customData = entry.customData;
    platformData = copyPlatformData(entry.platformData);
    return *this;
}

PasteboardCustomData::Entry& PasteboardCustomData::Entry::operator=(Entry&&) = default;

PasteboardCustomData::PasteboardCustomData() = default;
PasteboardCustomData::PasteboardCustomData(const PasteboardCustomData&) = default;
PasteboardCustomData::PasteboardCustomData(PasteboardCustomData&&) = default;
PasteboardCustomData::~PasteboardCustomData() = default;

PasteboardCustomData::PasteboardCustomData(String&& origin, Vector<Entry>&& data)
    : m_origin(WTFMove(origin))
    , m_data(WTFMove(data))
{
}

Ref<SharedBuffer> PasteboardCustomData::createSharedBuffer() const
{
    constexpr unsigned currentCustomDataSerializationVersion = 1;

    WTF::Persistence::Encoder encoder;
    encoder << currentCustomDataSerializationVersion;
    encoder << m_origin;
    encoder << sameOriginCustomStringData();
    encoder << orderedTypes();
    return SharedBuffer::create(encoder.span());
}

PasteboardCustomData PasteboardCustomData::fromPersistenceDecoder(WTF::Persistence::Decoder&& decoder)
{
    constexpr unsigned maxSupportedDataSerializationVersionNumber = 1;

    PasteboardCustomData result;
    std::optional<unsigned> version;
    decoder >> version;
    if (!version || *version > maxSupportedDataSerializationVersionNumber)
        return { };

    std::optional<String> origin;
    decoder >> origin;
    if (!origin)
        return { };
    result.m_origin = WTFMove(*origin);

    std::optional<UncheckedKeyHashMap<String, String>> sameOriginCustomStringData;
    decoder >> sameOriginCustomStringData;
    if (!sameOriginCustomStringData)
        return { };

    std::optional<Vector<String>> orderedTypes;
    decoder >> orderedTypes;
    if (!orderedTypes)
        return { };

    for (auto& type : *orderedTypes)
        result.writeStringInCustomData(type, sameOriginCustomStringData->get(type));

    return result;
}

PasteboardCustomData PasteboardCustomData::fromSharedBuffer(const SharedBuffer& buffer)
{
    return fromPersistenceDecoder(buffer.decoder());
}

void PasteboardCustomData::writeString(const String& type, const String& value)
{
    addOrMoveEntryToEnd(type).platformData = { value };
}

void PasteboardCustomData::writeData(const String& type, Ref<SharedBuffer>&& data)
{
    addOrMoveEntryToEnd(type).platformData = { WTFMove(data) };
}

void PasteboardCustomData::writeStringInCustomData(const String& type, const String& value)
{
    addOrMoveEntryToEnd(type).customData = value;
}

PasteboardCustomData::Entry& PasteboardCustomData::addOrMoveEntryToEnd(const String& type)
{
    auto index = m_data.findIf([&] (auto& entry) {
        return entry.type == type;
    });
    auto entry = index == notFound ? Entry(type) : m_data[index];
    if (index != notFound)
        m_data.remove(index);
    m_data.append(WTFMove(entry));
    return m_data.last();
}

void PasteboardCustomData::clear()
{
    m_data.clear();
}

void PasteboardCustomData::clear(const String& type)
{
    m_data.removeFirstMatching([&] (auto& entry) {
        return entry.type == type;
    });
}

PasteboardCustomData& PasteboardCustomData::operator=(const PasteboardCustomData& other)
{
    m_origin = other.origin();
    m_data = other.m_data;
    return *this;
}

Vector<String> PasteboardCustomData::orderedTypes() const
{
    return m_data.map([&] (auto& entry) {
        return entry.type;
    });
}

bool PasteboardCustomData::hasData() const
{
    return !m_data.isEmpty();
}

bool PasteboardCustomData::hasSameOriginCustomData() const
{
    return notFound != m_data.findIf([&] (auto& entry) {
        return !entry.customData.isNull();
    });
}

UncheckedKeyHashMap<String, String> PasteboardCustomData::sameOriginCustomStringData() const
{
    UncheckedKeyHashMap<String, String> customData;
    for (auto& entry : m_data)
        customData.set(entry.type, entry.customData);
    return customData;
}

RefPtr<SharedBuffer> PasteboardCustomData::readBuffer(const String& type) const
{
    for (auto& entry : m_data) {
        if (entry.type != type)
            continue;

        if (std::holds_alternative<Ref<SharedBuffer>>(entry.platformData))
            return std::get<Ref<SharedBuffer>>(entry.platformData).copyRef();

        return nullptr;
    }
    return nullptr;
}

String PasteboardCustomData::readString(const String& type) const
{
    for (auto& entry : m_data) {
        if (entry.type != type)
            continue;

        if (std::holds_alternative<String>(entry.platformData))
            return std::get<String>(entry.platformData);

        return { };
    }
    return { };
}

String PasteboardCustomData::readStringInCustomData(const String& type) const
{
    for (auto& entry : m_data) {
        if (entry.type == type)
            return entry.customData;
    }
    return { };
}

void PasteboardCustomData::forEachType(Function<void(const String&)>&& function) const
{
    for (auto& entry : m_data)
        function(entry.type);
}

void PasteboardCustomData::forEachPlatformString(Function<void(const String& type, const String& data)>&& function) const
{
    for (auto& entry : m_data) {
        if (!std::holds_alternative<String>(entry.platformData))
            continue;

        auto string = std::get<String>(entry.platformData);
        if (!string.isNull())
            function(entry.type, string);
    }
}

void PasteboardCustomData::forEachCustomString(Function<void(const String& type, const String& data)>&& function) const
{
    for (auto& entry : m_data) {
        if (!entry.customData.isNull())
            function(entry.type, entry.customData);
    }
}

void PasteboardCustomData::forEachPlatformStringOrBuffer(Function<void(const String& type, const std::variant<String, Ref<SharedBuffer>>& data)>&& function) const
{
    for (auto& entry : m_data) {
        auto& data = entry.platformData;
        if ((std::holds_alternative<String>(data) && !std::get<String>(data).isNull()) || std::holds_alternative<Ref<SharedBuffer>>(data))
            function(entry.type, data);
    }
}

} // namespace WebCore
