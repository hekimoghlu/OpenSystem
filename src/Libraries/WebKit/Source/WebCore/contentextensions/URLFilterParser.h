/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include <wtf/text/WTFString.h>

namespace WebCore {

namespace ContentExtensions {

class CombinedURLFilters;

class WEBCORE_EXPORT URLFilterParser {
public:
    enum ParseStatus {
        Ok,
        MatchesEverything,
        NonASCII,
        UnsupportedCharacterClass,
        BackReference,
        ForwardReference,
        MisplacedStartOfLine,
        WordBoundary,
        AtomCharacter,
        Group,
        Disjunction,
        MisplacedEndOfLine,
        EmptyPattern,
        YarrError,
        InvalidQuantifier,
    };
    static ASCIILiteral statusString(ParseStatus);
    explicit URLFilterParser(CombinedURLFilters&);
    ~URLFilterParser();
    ParseStatus addPattern(StringView pattern, bool patternIsCaseSensitive, uint64_t patternId);

private:
    CombinedURLFilters& m_combinedURLFilters;
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
