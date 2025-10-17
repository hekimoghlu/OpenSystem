/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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
#include "APIArray.h"

#include "APIString.h"

namespace API {

Ref<Array> Array::create()
{
    return create(Vector<RefPtr<Object>>());
}

Ref<Array> Array::create(Vector<RefPtr<Object>>&& elements)
{
    return adoptRef(*new Array(WTFMove(elements)));
}

Ref<Array> Array::createWithCapacity(size_t capacity)
{
    auto array = create(Vector<RefPtr<Object>>());
    array->m_elements.reserveInitialCapacity(capacity);
    return array;
}

Ref<Array> Array::createStringArray(const Vector<WTF::String>& strings)
{
    auto elements = strings.map([](auto& string) -> RefPtr<Object> {
        return API::String::create(string);
    });
    return create(WTFMove(elements));
}

Ref<Array> Array::createStringArray(const std::span<const WTF::String> strings)
{
    return create(WTF::map(strings, [] (auto string) -> RefPtr<Object> {
        return API::String::create(string);
    }));
}

Vector<WTF::String> Array::toStringVector()
{
    Vector<WTF::String> patternsVector;

    size_t size = this->size();
    if (!size)
        return patternsVector;

    patternsVector.reserveInitialCapacity(size);
    for (auto entry : elementsOfType<API::String>())
        patternsVector.append(entry->string());
    return patternsVector;
}

Ref<API::Array> Array::copy()
{
    size_t size = this->size();
    if (!size)
        return Array::create();

    Vector<RefPtr<Object>> elements = this->elements();
    return Array::create(WTFMove(elements));
}

} // namespace API
