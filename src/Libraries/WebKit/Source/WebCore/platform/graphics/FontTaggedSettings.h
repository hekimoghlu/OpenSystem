/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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

#include <array>
#include <wtf/ArgumentCoder.h>
#include <wtf/HashCountedSet.h>
#include <wtf/HashTraits.h>
#include <wtf/Hasher.h>
#include <wtf/Vector.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

using FontTag = std::array<char, 4>;

inline FontTag fontFeatureTag(std::span<const char, 5> nullTerminatedString)
{
    ASSERT(nullTerminatedString[4] == '\0');
    return { nullTerminatedString[0], nullTerminatedString[1], nullTerminatedString[2], nullTerminatedString[3] };
}

inline void add(Hasher& hasher, std::array<char, 4> array)
{
    uint32_t integer = (static_cast<uint8_t>(array[0]) << 24) | (static_cast<uint8_t>(array[1]) << 16) | (static_cast<uint8_t>(array[2]) << 8) | static_cast<uint8_t>(array[3]);
    add(hasher, integer);
}

struct FourCharacterTagHash {
    static unsigned hash(FontTag characters) { return computeHash(characters); }
    static bool equal(FontTag a, FontTag b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = true;
};

struct FourCharacterTagHashTraits : HashTraits<FontTag> {
    static const bool emptyValueIsZero = true;
    static void constructDeletedValue(FontTag& slot) { new (NotNull, std::addressof(slot)) FontTag({{ ff, ff, ff, ff }}); }
    static bool isDeletedValue(FontTag value) { return value == FontTag({{ ff, ff, ff, ff }}); }

private:
    static constexpr char ff = static_cast<char>(0xFF);
};

template <typename T>
class FontTaggedSetting {
private:
    friend struct IPC::ArgumentCoder<FontTaggedSetting, void>;
public:
    FontTaggedSetting() = delete;
    FontTaggedSetting(FontTag, T value);

    friend bool operator==(const FontTaggedSetting&, const FontTaggedSetting&) = default;
    bool operator<(const FontTaggedSetting<T>& other) const;

    FontTag tag() const { return m_tag; }
    T value() const { return m_value; }
    bool enabled() const { return value(); }

private:
    FontTag m_tag;
    T m_value;
};

template <typename T>
FontTaggedSetting<T>::FontTaggedSetting(FontTag tag, T value)
    : m_tag(tag)
    , m_value(value)
{
}

template<typename T> void add(Hasher& hasher, const FontTaggedSetting<T>& setting)
{
    add(hasher, setting.tag(), setting.value());
}

template <typename T>
class FontTaggedSettings {
private:
    friend struct IPC::ArgumentCoder<FontTaggedSettings, void>;
public:
    using Setting = FontTaggedSetting<T>;

    void insert(FontTaggedSetting<T>&&);
    friend bool operator==(const FontTaggedSettings&, const FontTaggedSettings&) = default;

    bool isEmpty() const { return !size(); }
    size_t size() const { return m_list.size(); }
    const FontTaggedSetting<T>& operator[](int index) const { return m_list[index]; }
    const FontTaggedSetting<T>& at(size_t index) const { return m_list.at(index); }

    typename Vector<FontTaggedSetting<T>>::const_iterator begin() const { return m_list.begin(); }
    typename Vector<FontTaggedSetting<T>>::const_iterator end() const { return m_list.end(); }

    unsigned hash() const;

private:
    Vector<FontTaggedSetting<T>> m_list;
};

template <typename T>
void FontTaggedSettings<T>::insert(FontTaggedSetting<T>&& feature)
{
    // This vector will almost always have 0 or 1 items in it. Don't bother with the overhead of a binary search or a hash set.
    // We keep the vector sorted alphabetically and replace any pre-existing value for a given tag.
    size_t i;
    for (i = 0; i < m_list.size(); ++i) {
        if (m_list[i].tag() < feature.tag())
            continue;
        if (m_list[i].tag() == feature.tag())
            m_list.remove(i);
        break;
    }
    m_list.insert(i, WTFMove(feature));
}

using FontFeature = FontTaggedSetting<int>;
using FontFeatureSettings = FontTaggedSettings<int>;
using FontVariationSettings = FontTaggedSettings<float>;

TextStream& operator<<(TextStream&, const FontTaggedSettings<int>&);
TextStream& operator<<(TextStream&, const FontTaggedSettings<float>&);

}
