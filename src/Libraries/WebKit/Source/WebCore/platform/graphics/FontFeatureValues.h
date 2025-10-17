/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

#include <wtf/HashMap.h>
#include <wtf/Hasher.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

using FontFeatureValuesTag = std::pair<String, Vector<unsigned>>;

enum class FontFeatureValuesType {
    Styleset,
    Stylistic,
    CharacterVariant,
    Swash,
    Ornaments,
    Annotation
};

class FontFeatureValues : public RefCounted<FontFeatureValues> {
public:
    using Tags = UncheckedKeyHashMap<String, Vector<unsigned>>;
    static Ref<FontFeatureValues> create() { return adoptRef(*new FontFeatureValues()); }
    virtual ~FontFeatureValues() = default;

    bool isEmpty() const
    {
        return m_styleset.isEmpty() 
            && m_stylistic.isEmpty() 
            && m_characterVariant.isEmpty() 
            && m_swash.isEmpty()
            && m_ornaments.isEmpty()
            && m_annotation.isEmpty();
    }
    
    bool operator==(const FontFeatureValues& other) const
    {
        return m_styleset == other.styleset()
            && m_stylistic == other.stylistic()
            && m_characterVariant == other.characterVariant()
            && m_swash == other.swash()
            && m_ornaments == other.ornaments()
            && m_annotation == other.annotation();
    }
    
    const Tags& styleset() const
    {
        return m_styleset;
    }

    Tags& styleset()
    {
        return m_styleset;
    }

    const Tags& stylistic() const
    {
        return m_stylistic;
    }

    Tags& stylistic()
    {
        return m_stylistic;
    }

    const Tags& characterVariant() const
    {
        return m_characterVariant;
    }

    Tags& characterVariant()
    {
        return m_characterVariant;
    }

    const Tags& swash() const
    {
        return m_swash;
    }

    Tags& swash()
    {
        return m_swash;
    }

    const Tags& ornaments() const
    {
        return m_ornaments;
    }

    Tags& ornaments()
    {
        return m_ornaments;
    }

    const Tags& annotation() const
    {
        return m_annotation;
    }

    Tags& annotation()
    {
        return m_annotation;
    }

    friend void add(Hasher&, const FontFeatureValues&);
    friend WTF::TextStream& operator<<(WTF::TextStream&, const FontFeatureValues&);
    void updateOrInsert(const FontFeatureValues&);
    void updateOrInsertForType(FontFeatureValuesType, const Vector<FontFeatureValuesTag>&);

private:
    FontFeatureValues() = default;
    Tags m_styleset;
    Tags m_stylistic;
    Tags m_characterVariant;
    Tags m_swash;
    Tags m_ornaments;
    Tags m_annotation;
};

} // namespace WebCore
