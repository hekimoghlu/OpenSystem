/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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
#include "APIDictionary.h"

#include "APIArray.h"
#include "APIString.h"

namespace API {

Ref<Dictionary> Dictionary::create()
{
    return create({ });
}

Ref<Dictionary> Dictionary::createWithCapacity(size_t capacity)
{
    auto dictionary = create();
    dictionary->m_map.reserveInitialCapacity(capacity);
    return dictionary;
}

Ref<Dictionary> Dictionary::create(MapType&& map)
{
    return adoptRef(*new Dictionary(WTFMove(map)));
}

Dictionary::Dictionary(MapType&& map)
    : m_map(WTFMove(map))
{
}

Dictionary::~Dictionary() = default;

Ref<Array> Dictionary::keys() const
{
    if (m_map.isEmpty())
        return API::Array::create();

    auto keys = WTF::map(m_map, [](auto& entry) -> RefPtr<API::Object> {
        return API::String::create(entry.key);
    });
    return API::Array::create(WTFMove(keys));
}

bool Dictionary::add(const WTF::String& key, RefPtr<API::Object>&& item)
{
    MapType::AddResult result = m_map.add(key, WTFMove(item));
    return result.isNewEntry;
}

bool Dictionary::set(const WTF::String& key, RefPtr<API::Object>&& item)
{
    MapType::AddResult result = m_map.set(key, WTFMove(item));
    return result.isNewEntry;
}

void Dictionary::remove(const WTF::String& key)
{
    m_map.remove(key);
}

} // namespace API
