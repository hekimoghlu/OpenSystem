/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#include "ContentSecurityPolicyMediaListDirective.h"

#include "ContentSecurityPolicy.h"
#include "ContentSecurityPolicyDirectiveList.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/ParsingUtilities.h>
#include <wtf/text/StringHash.h>
#include <wtf/text/StringParsingBuffer.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ContentSecurityPolicyMediaListDirective);

template<typename CharacterType> static bool isMediaTypeCharacter(CharacterType c)
{
    return !isUnicodeCompatibleASCIIWhitespace(c) && c != '/';
}

ContentSecurityPolicyMediaListDirective::ContentSecurityPolicyMediaListDirective(const ContentSecurityPolicyDirectiveList& directiveList, const String& name, const String& value)
    : ContentSecurityPolicyDirective(directiveList, name, value)
{
    parse(value);
}

bool ContentSecurityPolicyMediaListDirective::allows(const String& type) const
{
    return m_pluginTypes.contains(type);
}

void ContentSecurityPolicyMediaListDirective::parse(const String& value)
{
    // 'plugin-types ____;' OR 'plugin-types;'
    if (value.isEmpty()) {
        directiveList().policy().reportInvalidPluginTypes(value);
        return;
    }

    readCharactersForParsing(value, [&](auto buffer) {
        while (buffer.hasCharactersRemaining()) {
            // _____ OR _____mime1/mime1
            // ^        ^
            skipWhile<isUnicodeCompatibleASCIIWhitespace>(buffer);
            if (buffer.atEnd())
                return;

            // mime1/mime1 mime2/mime2
            // ^
            auto begin = buffer.position();
            if (!skipExactly<isMediaTypeCharacter>(buffer)) {
                skipWhile<isNotASCIISpace>(buffer);
                directiveList().policy().reportInvalidPluginTypes(String({ begin, buffer.position() }));
                continue;
            }
            skipWhile<isMediaTypeCharacter>(buffer);

            // mime1/mime1 mime2/mime2
            //      ^
            if (!skipExactly(buffer, '/')) {
                skipWhile<isNotASCIISpace>(buffer);
                directiveList().policy().reportInvalidPluginTypes(String({ begin, buffer.position() }));
                continue;
            }

            // mime1/mime1 mime2/mime2
            //       ^
            if (!skipExactly<isMediaTypeCharacter>(buffer)) {
                skipWhile<isNotASCIISpace>(buffer);
                directiveList().policy().reportInvalidPluginTypes(String({ begin, buffer.position() }));
                continue;
            }
            skipWhile<isMediaTypeCharacter>(buffer);

            // mime1/mime1 mime2/mime2 OR mime1/mime1  OR mime1/mime1/error
            //            ^                          ^               ^
            if (buffer.hasCharactersRemaining() && isNotASCIISpace(*buffer)) {
                skipWhile<isNotASCIISpace>(buffer);
                directiveList().policy().reportInvalidPluginTypes(String({ begin, buffer.position() }));
                continue;
            }
            m_pluginTypes.add(String({ begin, buffer.position() }));

            ASSERT(buffer.atEnd() || isUnicodeCompatibleASCIIWhitespace(*buffer));
        }
    });
}

} // namespace WebCore
