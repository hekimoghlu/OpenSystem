/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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
#import "config.h"
#import "StringUtilities.h"

#import <JavaScriptCore/RegularExpression.h>
#import <wtf/text/StringBuilder.h>

namespace WebCore {
    
static String wildcardRegexPatternString(const String& string)
{
    String metaCharacters = ".|+?()[]{}^$"_s;
    UChar escapeCharacter = '\\';
    UChar wildcardCharacter = '*';
    
    StringBuilder stringBuilder;
    
    stringBuilder.append('^');
    for (unsigned i = 0; i < string.length(); i++) {
        auto character = string[i];
        if (metaCharacters.contains(character) || character == escapeCharacter)
            stringBuilder.append(escapeCharacter);
        else if (character == wildcardCharacter)
            stringBuilder.append('.');
        
        stringBuilder.append(character);
    }
    stringBuilder.append('$');
        
    return stringBuilder.toString();
}
    
bool stringMatchesWildcardString(const String& string, const String& wildcardString)
{
    return JSC::Yarr::RegularExpression(wildcardRegexPatternString(wildcardString), { JSC::Yarr::Flags::IgnoreCase }).match(string) != -1;
}

}
