/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 6, 2024.
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

#include <unicode/umachine.h>

namespace WebCore {

// Optimize packing since there are over 2000 of these.
struct HTMLEntityTableEntry {
    const char* nameCharacters() const;
    unsigned nameLength() const { return nameLengthExcludingSemicolon + nameIncludesTrailingSemicolon; }

    unsigned firstCharacter : 21; // All Unicode characters fit in 21 bits.
    unsigned optionalSecondCharacter : 16; // Two-character sequences are all BMP characters.
    unsigned nameCharactersOffset : 14;
    unsigned nameLengthExcludingSemicolon : 5;
    unsigned nameIncludesTrailingSemicolon : 1;
};

class HTMLEntityTable {
public:
    static const HTMLEntityTableEntry* firstEntry();
    static const HTMLEntityTableEntry* lastEntry();

    static const HTMLEntityTableEntry* firstEntryStartingWith(UChar);
    static const HTMLEntityTableEntry* lastEntryStartingWith(UChar);
};

} // namespace WebCore
