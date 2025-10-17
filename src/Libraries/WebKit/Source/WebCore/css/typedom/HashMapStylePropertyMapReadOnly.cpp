/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#include "HashMapStylePropertyMapReadOnly.h"

#include "CSSPropertyParser.h"

namespace WebCore {

Ref<HashMapStylePropertyMapReadOnly> HashMapStylePropertyMapReadOnly::create(UncheckedKeyHashMap<AtomString, RefPtr<CSSValue>>&& map)
{
    return adoptRef(*new HashMapStylePropertyMapReadOnly(WTFMove(map)));
}

HashMapStylePropertyMapReadOnly::HashMapStylePropertyMapReadOnly(UncheckedKeyHashMap<AtomString, RefPtr<CSSValue>>&& map)
    : m_map(WTFMove(map))
{
}

HashMapStylePropertyMapReadOnly::~HashMapStylePropertyMapReadOnly() = default;

RefPtr<CSSValue> HashMapStylePropertyMapReadOnly::propertyValue(CSSPropertyID propertyID) const
{
    return m_map.get(nameString(propertyID));
}

String HashMapStylePropertyMapReadOnly::shorthandPropertySerialization(CSSPropertyID) const
{
    // FIXME: Not supported.
    return { };
}

RefPtr<CSSValue> HashMapStylePropertyMapReadOnly::customPropertyValue(const AtomString& property) const
{
    return m_map.get(property);
}

unsigned HashMapStylePropertyMapReadOnly::size() const
{
    return m_map.size();
}

auto HashMapStylePropertyMapReadOnly::entries(ScriptExecutionContext* context) const -> Vector<StylePropertyMapEntry>
{
    auto* document = context ? documentFromContext(*context) : nullptr;
    if (!document)
        return { };

    return WTF::map(m_map, [&](auto& entry) -> StylePropertyMapEntry {
        auto& [propertyName, cssValue] = entry;
        return makeKeyValuePair(propertyName,  Vector<RefPtr<CSSStyleValue>> { reifyValue(cssValue.get(), cssPropertyID(propertyName), *document) });
    });
}

} // namespace WebCore
