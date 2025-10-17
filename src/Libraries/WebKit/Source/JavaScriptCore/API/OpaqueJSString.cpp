/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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
#include "OpaqueJSString.h"

#include "Identifier.h"
#include "IdentifierInlines.h"
#include <wtf/MallocSpan.h>
#include <wtf/text/StringView.h>

using namespace JSC;

RefPtr<OpaqueJSString> OpaqueJSString::tryCreate(const String& string)
{
    if (string.isNull())
        return nullptr;

    return adoptRef(new OpaqueJSString(string));
}

RefPtr<OpaqueJSString> OpaqueJSString::tryCreate(String&& string)
{
    if (string.isNull())
        return nullptr;

    return adoptRef(new OpaqueJSString(WTFMove(string)));
}

OpaqueJSString::~OpaqueJSString()
{
    // m_characters is put in a local here to avoid an extra atomic load.
    UChar* characters = m_characters;
    if (!characters)
        return;

    if (!m_string.is8Bit() && m_string.span16().data() == characters)
        return;

    fastFree(characters);
}

String OpaqueJSString::string() const
{
    // Return a copy of the wrapped string, because the caller may make it an Identifier.
    return m_string.isolatedCopy();
}

Identifier OpaqueJSString::identifier(VM* vm) const
{
    if (m_string.isNull())
        return Identifier();
    if (m_string.isEmpty())
        return Identifier(Identifier::EmptyIdentifier);
    if (m_string.is8Bit())
        return Identifier::fromString(*vm, m_string.span8());
    return Identifier::fromString(*vm, m_string.span16());
}

const UChar* OpaqueJSString::characters()
{
    // m_characters is put in a local here to avoid an extra atomic load.
    UChar* characters = m_characters;
    if (characters)
        return characters;

    if (m_string.isNull())
        return nullptr;

    auto newCharacters = MallocSpan<UChar>::malloc(m_string.length() * sizeof(UChar));
    StringView { m_string }.getCharacters(newCharacters.mutableSpan());

    if (!m_characters.compare_exchange_strong(characters, newCharacters.mutableSpan().data()))
        return characters;

    return newCharacters.leakSpan().data();
}

bool OpaqueJSString::equal(const OpaqueJSString* a, const OpaqueJSString* b)
{
    if (a == b)
        return true;

    if (!a || !b)
        return false;

    return a->m_string == b->m_string;
}
