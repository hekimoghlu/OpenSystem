/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 27, 2025.
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
#include "ContentSecurityPolicyTrustedTypesDirective.h"

#include "ContentSecurityPolicy.h"
#include "ContentSecurityPolicyDirectiveList.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/ParsingUtilities.h>
#include <wtf/text/StringCommon.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ContentSecurityPolicyTrustedTypesDirective);

template<typename CharacterType> static bool isTrustedTypesNone(StringParsingBuffer<CharacterType> buffer)
{
    skipWhile<isASCIIWhitespace>(buffer);

    if (!skipExactlyIgnoringASCIICase(buffer, "'none'"_s))
        return false;

    skipWhile<isASCIIWhitespace>(buffer);

    return buffer.atEnd();
}

template<typename CharacterType> static bool isTrustedTypeCharacter(CharacterType c)
{
    return !isASCIIWhitespace(c);
}

template<typename CharacterType> static bool isPolicyNameCharacter(CharacterType c)
{
    return isASCIIAlphanumeric(c) || c == '-' || c == '#' || c == '=' || c == '_' || c == '/' || c == '@' || c == '.' || c == '%';
}

ContentSecurityPolicyTrustedTypesDirective::ContentSecurityPolicyTrustedTypesDirective(const ContentSecurityPolicyDirectiveList& directiveList, const String& name, const String& value)
    : ContentSecurityPolicyDirective(directiveList, name, value)
{
    parse(value);
}

bool ContentSecurityPolicyTrustedTypesDirective::allows(const String& value, bool isDuplicate, AllowTrustedTypePolicy& details) const
{
    auto invalidPolicy = value.find([](UChar ch) {
        return !isPolicyNameCharacter(ch);
    });

    if (isDuplicate && !m_allowDuplicates)
        details = AllowTrustedTypePolicy::DisallowedDuplicateName;
    else if (isDuplicate && value == "default"_s)
        details = AllowTrustedTypePolicy::DisallowedDuplicateName;
    else if (invalidPolicy != notFound)
        details = AllowTrustedTypePolicy::DisallowedName;
    else if (!(m_allowAny || m_list.contains(value)))
        details = AllowTrustedTypePolicy::DisallowedName;
    else
        details = AllowTrustedTypePolicy::Allowed;

    return details == AllowTrustedTypePolicy::Allowed;
}

void ContentSecurityPolicyTrustedTypesDirective::parse(const String& value)
{
    // 'trusted-types;'
    if (value.isEmpty())
        return;

    readCharactersForParsing(value, [&](auto buffer) {
        if (isTrustedTypesNone(buffer))
            return;

        while (buffer.hasCharactersRemaining()) {
            skipWhile<isASCIIWhitespace>(buffer);
            if (buffer.atEnd())
                return;

            auto beginPolicy = buffer.position();
            skipWhile<isTrustedTypeCharacter>(buffer);

            StringParsingBuffer policyBuffer(std::span(beginPolicy, buffer.position()));

            if (skipExactlyIgnoringASCIICase(policyBuffer, "'allow-duplicates'"_s)) {
                m_allowDuplicates = true;
                continue;
            }

            if (skipExactlyIgnoringASCIICase(policyBuffer, "'none'"_s)) {
                directiveList().policy().reportInvalidTrustedTypesNoneKeyword();
                continue;
            }

            if (skipExactly(policyBuffer, '*')) {
                m_allowAny = true;
                continue;
            }

            if (skipExactly<isPolicyNameCharacter>(policyBuffer)) {
                auto policy = String({ beginPolicy, buffer.position() });
                m_list.add(policy);
            } else {
                auto policy = String({ beginPolicy, buffer.position() });
                directiveList().policy().reportInvalidTrustedTypesPolicy(policy);
                return;
            }

            ASSERT(buffer.atEnd() || isASCIIWhitespace(*buffer));
        }
    });
}

} // namespace WebCore
