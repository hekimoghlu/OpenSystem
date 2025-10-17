/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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
#include "DatasetDOMStringMap.h"

#include "ElementInlines.h"
#include <wtf/ASCIICType.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/AtomString.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DatasetDOMStringMap);

static bool isValidAttributeName(const String& name)
{
    if (!name.startsWith("data-"_s))
        return false;

    unsigned length = name.length();
    for (unsigned i = 5; i < length; ++i) {
        if (isASCIIUpper(name[i]))
            return false;
    }

    return true;
}

static String convertAttributeNameToPropertyName(const String& name)
{
    StringBuilder stringBuilder;

    unsigned length = name.length();
    for (unsigned i = 5; i < length; ++i) {
        UChar character = name[i];
        if (character != '-')
            stringBuilder.append(character);
        else {
            if ((i + 1 < length) && isASCIILower(name[i + 1])) {
                stringBuilder.append(toASCIIUpper(name[i + 1]));
                ++i;
            } else
                stringBuilder.append(character);
        }
    }

    return stringBuilder.toString();
}

static bool isValidPropertyName(const String& name)
{
    unsigned length = name.length();
    for (unsigned i = 0; i < length; ++i) {
        if (name[i] == '-' && (i + 1 < length) && isASCIILower(name[i + 1]))
            return false;
    }
    return true;
}

template<typename CharacterType>
static inline AtomString convertPropertyNameToAttributeName(const StringImpl& name)
{
    const CharacterType dataPrefix[] = { 'd', 'a', 't', 'a', '-' };

    Vector<CharacterType, 32> buffer;

    unsigned length = name.length();
    buffer.reserveInitialCapacity(std::size(dataPrefix) + length);

    buffer.append(std::span { dataPrefix });

    auto characters = name.span<CharacterType>();
    for (auto character : characters) {
        if (isASCIIUpper(character)) {
            buffer.append('-');
            buffer.append(toASCIILower(character));
        } else
            buffer.append(character);
    }
    return buffer.span();
}

static AtomString convertPropertyNameToAttributeName(const String& name)
{
    if (name.isNull())
        return nullAtom();

    StringImpl* nameImpl = name.impl();
    if (nameImpl->is8Bit())
        return convertPropertyNameToAttributeName<LChar>(*nameImpl);
    return convertPropertyNameToAttributeName<UChar>(*nameImpl);
}

void DatasetDOMStringMap::ref()
{
    m_element->ref();
}

void DatasetDOMStringMap::deref()
{
    m_element->deref();
}

bool DatasetDOMStringMap::isSupportedPropertyName(const String& propertyName) const
{
    Ref element = m_element.get();
    if (!element->hasAttributes())
        return false;

    auto attributes = element->attributes();
    if (attributes.size() == 1) {
        // Avoid creating AtomString when there is only one attribute.
        auto& attribute = attributes[0];
        if (convertAttributeNameToPropertyName(attribute.localName()) == propertyName)
            return true;
    } else {
        auto attributeName = convertPropertyNameToAttributeName(propertyName);
        for (auto& attribute : attributes) {
            if (attribute.localName() == attributeName)
                return true;
        }
    }
    
    return false;
}

Vector<String> DatasetDOMStringMap::supportedPropertyNames() const
{
    Vector<String> names;

    Ref element = m_element.get();
    if (!element->hasAttributes())
        return names;

    for (auto& attribute : element->attributes()) {
        if (isValidAttributeName(attribute.localName()))
            names.append(convertAttributeNameToPropertyName(attribute.localName()));
    }

    return names;
}

const AtomString* DatasetDOMStringMap::item(const String& propertyName) const
{
    Ref element = m_element.get();
    if (element->hasAttributes()) {
        auto attributes = element->attributes();

        if (attributes.size() == 1) {
            // Avoid creating AtomString when there is only one attribute.
            auto& attribute = attributes[0];
            if (convertAttributeNameToPropertyName(attribute.localName()) == propertyName)
                return &attribute.value();
        } else {
            AtomString attributeName = convertPropertyNameToAttributeName(propertyName);
            for (auto& attribute : attributes) {
                if (attribute.localName() == attributeName)
                    return &attribute.value();
            }
        }
    }

    return nullptr;
}

String DatasetDOMStringMap::namedItem(const AtomString& name) const
{
    if (const auto* value = item(name))
        return *value;
    return String { };
}

ExceptionOr<void> DatasetDOMStringMap::setNamedItem(const String& name, const AtomString& value)
{
    if (!isValidPropertyName(name))
        return Exception { ExceptionCode::SyntaxError };
    return protectedElement()->setAttribute(convertPropertyNameToAttributeName(name), value);
}

bool DatasetDOMStringMap::deleteNamedProperty(const String& name)
{
    return protectedElement()->removeAttribute(convertPropertyNameToAttributeName(name));
}

Ref<Element> DatasetDOMStringMap::protectedElement() const
{
    return m_element.get();
}

} // namespace WebCore
