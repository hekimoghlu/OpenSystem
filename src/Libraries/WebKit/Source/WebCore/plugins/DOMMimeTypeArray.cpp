/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 6, 2024.
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
#include "DOMMimeTypeArray.h"

#include "DOMMimeType.h"
#include "Navigator.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DOMMimeTypeArray);

Ref<DOMMimeTypeArray> DOMMimeTypeArray::create(Navigator& navigator, Vector<Ref<DOMMimeType>>&& types)
{
    return adoptRef(*new DOMMimeTypeArray(navigator, WTFMove(types)));
}

DOMMimeTypeArray::DOMMimeTypeArray(Navigator& navigator, Vector<Ref<DOMMimeType>>&& types)
    : m_navigator(navigator)
    , m_types(WTFMove(types))
{
}

DOMMimeTypeArray::~DOMMimeTypeArray() = default;

unsigned DOMMimeTypeArray::length() const
{
    return m_types.size();
}

RefPtr<DOMMimeType> DOMMimeTypeArray::item(unsigned index)
{
    if (index >= m_types.size())
        return nullptr;
    return m_types[index].ptr();
}

RefPtr<DOMMimeType> DOMMimeTypeArray::namedItem(const AtomString& propertyName)
{
    for (auto& type : m_types) {
        if (type->type() == propertyName)
            return type.ptr();
    }
    return nullptr;
}

bool DOMMimeTypeArray::isSupportedPropertyName(const AtomString& propertyName) const
{
    return m_types.containsIf([&](auto& type) { return type->type() == propertyName; });
}

Vector<AtomString> DOMMimeTypeArray::supportedPropertyNames() const
{
    return m_types.map([](auto& type) -> AtomString {
        return type->type();
    });
}

} // namespace WebCore
