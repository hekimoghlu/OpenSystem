/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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

#include "APIObject.h"
#include <wtf/HashMap.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/WTFString.h>

namespace API {

class Array;

class Dictionary final : public ObjectImpl<Object::Type::Dictionary> {
public:
    using MapType = HashMap<WTF::String, RefPtr<Object>>;

    static Ref<Dictionary> create();
    static Ref<Dictionary> createWithCapacity(size_t);
    static Ref<Dictionary> create(MapType&&);

    virtual ~Dictionary();

    template<typename T>
    T* get(const WTF::String& key) const
    {
        RefPtr<Object> item = m_map.get(key);
        if (!item)
            return nullptr;

        if (item->type() != T::APIType)
            return nullptr;

        return static_cast<T*>(item.get());
    }

    Object* get(const WTF::String& key) const
    {
        return m_map.get(key);
    }

    Object* get(const WTF::String& key, bool& exists) const
    {
        auto it = m_map.find(key);
        exists = it != m_map.end();
        if (!exists)
            return nullptr;
        
        return it->value.get();
    }

    Ref<Array> keys() const;

    bool add(const WTF::String& key, RefPtr<Object>&&);
    bool set(const WTF::String& key, RefPtr<Object>&&);
    void remove(const WTF::String& key);

    size_t size() const { return m_map.size(); }

    const MapType& map() const { return m_map; }

private:
    explicit Dictionary(MapType&&);

    MapType m_map;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(Dictionary);
