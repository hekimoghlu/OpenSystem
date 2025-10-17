/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include "FontFeatureValues.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

void add(Hasher& hasher, const FontFeatureValues& fontFeatureValues)
{
    auto hashTags = [&hasher](const auto& tags) {
        add(hasher, tags.isEmpty());
        for (const auto& tag : tags)
            add(hasher, tag.key, tag.value);
    };
    hashTags(fontFeatureValues.m_styleset);
    hashTags(fontFeatureValues.m_stylistic);
    hashTags(fontFeatureValues.m_characterVariant);
    hashTags(fontFeatureValues.m_swash);
    hashTags(fontFeatureValues.m_ornaments);
    hashTags(fontFeatureValues.m_annotation);
}

void FontFeatureValues::updateOrInsert(const FontFeatureValues& other)
{
    if (this == &other)
        return;
    
    auto updateOrInsertTags = [](auto& into, const auto& tags) {
        for (const auto& tag : tags)
            into.set(tag.key, tag.value);
    };
    updateOrInsertTags(m_styleset, other.styleset());
    updateOrInsertTags(m_stylistic, other.stylistic());
    updateOrInsertTags(m_characterVariant, other.characterVariant());
    updateOrInsertTags(m_swash, other.swash());
    updateOrInsertTags(m_ornaments, other.ornaments());
    updateOrInsertTags(m_annotation, other.annotation());
}

void FontFeatureValues::updateOrInsertForType(FontFeatureValuesType type, const Vector<FontFeatureValuesTag>& tags)
{
    auto updateOrInsertTags = [](auto& into, const Vector<FontFeatureValuesTag>& tags) {
        for (const FontFeatureValuesTag& tag : tags)
            into.set(tag.first, tag.second);
    };
    switch (type) {
    case FontFeatureValuesType::Styleset:
        updateOrInsertTags(m_styleset, tags);
        break;
    case FontFeatureValuesType::Stylistic:
        updateOrInsertTags(m_stylistic, tags);
        break;
    case FontFeatureValuesType::CharacterVariant:
        updateOrInsertTags(m_characterVariant, tags);
        break;
    case FontFeatureValuesType::Swash:
        updateOrInsertTags(m_swash, tags);
        break;
    case FontFeatureValuesType::Ornaments:
        updateOrInsertTags(m_ornaments, tags);
        break;
    case FontFeatureValuesType::Annotation:
        updateOrInsertTags(m_annotation, tags);
        break;
    }
    
}

WTF::TextStream& operator<<(WTF::TextStream& ts, const FontFeatureValues& fontFeatureValues)
{
    auto printTags = [&ts](const auto& name, const auto& tags) {
        if (tags.isEmpty())
            return;

        ts << "{" << name << ": ";
        for (const auto& tag : tags)
            ts << "{" << tag.key << ":" << tag.value << "} ";
        ts << "}";
    };
    printTags("styleset", fontFeatureValues.m_styleset);
    printTags("stylistic", fontFeatureValues.m_stylistic);
    printTags("characterVariant", fontFeatureValues.m_characterVariant);
    printTags("swash", fontFeatureValues.m_swash);
    printTags("ornaments", fontFeatureValues.m_ornaments);
    printTags("annotation", fontFeatureValues.m_annotation);
    return ts;  
}

} // namespace WebCore
