/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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

#include <limits>
#include <wtf/Vector.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace JSC {

class TemplateObjectDescriptorTable;

class TemplateObjectDescriptor : public RefCounted<TemplateObjectDescriptor> {
public:
    typedef Vector<String, 4> StringVector;
    typedef Vector<std::optional<String>, 4> OptionalStringVector;

    enum DeletedValueTag { DeletedValue };
    TemplateObjectDescriptor(DeletedValueTag);
    enum EmptyValueTag { EmptyValue };
    TemplateObjectDescriptor(EmptyValueTag);

    bool isDeletedValue() const { return m_rawStrings.isEmpty() && m_hash == std::numeric_limits<unsigned>::max(); }

    bool isEmptyValue() const { return m_rawStrings.isEmpty() && !m_hash; }

    unsigned hash() const { return m_hash; }

    const StringVector& rawStrings() const { return m_rawStrings; }
    const OptionalStringVector& cookedStrings() const { return m_cookedStrings; }

    bool operator==(const TemplateObjectDescriptor& other) const { return m_hash == other.m_hash && m_rawStrings == other.m_rawStrings; }

    struct Hasher {
        static unsigned hash(const TemplateObjectDescriptor& key) { return key.hash(); }
        static bool equal(const TemplateObjectDescriptor& a, const TemplateObjectDescriptor& b) { return a == b; }
        static constexpr bool safeToCompareToEmptyOrDeleted = false;
    };

    static unsigned calculateHash(const StringVector& rawStrings);
    ~TemplateObjectDescriptor();

    static Ref<TemplateObjectDescriptor> create(StringVector&& rawStrings, OptionalStringVector&& cookedStrings)
    {
        return adoptRef(*new TemplateObjectDescriptor(WTFMove(rawStrings), WTFMove(cookedStrings)));
    }

private:
    TemplateObjectDescriptor(StringVector&& rawStrings, OptionalStringVector&& cookedStrings);

    StringVector m_rawStrings;
    OptionalStringVector m_cookedStrings;
    unsigned m_hash { 0 };
};

inline TemplateObjectDescriptor::TemplateObjectDescriptor(StringVector&& rawStrings, OptionalStringVector&& cookedStrings)
    : m_rawStrings(WTFMove(rawStrings))
    , m_cookedStrings(WTFMove(cookedStrings))
    , m_hash(calculateHash(m_rawStrings))
{
}

inline TemplateObjectDescriptor::TemplateObjectDescriptor(DeletedValueTag)
    : m_hash(std::numeric_limits<unsigned>::max())
{
}

inline TemplateObjectDescriptor::TemplateObjectDescriptor(EmptyValueTag)
    : m_hash(0)
{
}

inline unsigned TemplateObjectDescriptor::calculateHash(const StringVector& rawStrings)
{
    SuperFastHash hasher;
    for (const String& string : rawStrings) {
        if (string.is8Bit())
            hasher.addCharacters(string.span8());
        else
            hasher.addCharacters(string.span16());
    }
    return hasher.hash();
}

} // namespace JSC

namespace WTF {
template<typename> struct DefaultHash;

template<> struct DefaultHash<JSC::TemplateObjectDescriptor> : JSC::TemplateObjectDescriptor::Hasher { };

template<> struct HashTraits<JSC::TemplateObjectDescriptor> : CustomHashTraits<JSC::TemplateObjectDescriptor> {
};

} // namespace WTF
