/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 4, 2022.
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
#include "SmartReplace.h"

#include <CoreFoundation/CFCharacterSet.h>
#include <CoreFoundation/CFString.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RetainPtr.h>

namespace WebCore {

static CFMutableCharacterSetRef getSmartSet(bool isPreviousCharacter)
{
    static NeverDestroyed<RetainPtr<CFMutableCharacterSetRef>> preSmartSet;
    static NeverDestroyed<RetainPtr<CFMutableCharacterSetRef>> postSmartSet;
    RetainPtr<CFMutableCharacterSetRef>& smartSet = isPreviousCharacter ? preSmartSet.get() : postSmartSet.get();
    if (!smartSet) {
        smartSet = adoptCF(CFCharacterSetCreateMutable(kCFAllocatorDefault));
        CFCharacterSetAddCharactersInString(smartSet.get(), isPreviousCharacter ? CFSTR("([\"\'#$/-`{") : CFSTR(")].,;:?\'!\"%*-/}"));
        CFCharacterSetUnion(smartSet.get(), CFCharacterSetGetPredefined(kCFCharacterSetWhitespaceAndNewline));

        // Adding CJK ranges
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0x1100, 256)); // Hangul Jamo (0x1100 - 0x11FF)
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0x2E80, 352)); // CJK & Kangxi Radicals (0x2E80 - 0x2FDF)
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0x2FF0, 464)); // Ideograph Descriptions, CJK Symbols, Hiragana, Katakana, Bopomofo, Hangul Compatibility Jamo, Kanbun, & Bopomofo Ext (0x2FF0 - 0x31BF)
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0x3200, 29392)); // Enclosed CJK, CJK Ideographs (Uni Han & Ext A), & Yi (0x3200 - 0xA4CF)
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0xAC00, 11183)); // Hangul Syllables (0xAC00 - 0xD7AF)
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0xF900, 352)); // CJK Compatibility Ideographs (0xF900 - 0xFA5F)
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0xFE30, 32)); // CJK Compatibility From (0xFE30 - 0xFE4F)
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0xFF00, 240)); // Half/Full Width Form (0xFF00 - 0xFFEF)
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0x20000, 0xA6D7)); // CJK Ideograph Exntension B
        CFCharacterSetAddCharactersInRange(smartSet.get(), CFRangeMake(0x2F800, 0x021E)); // CJK Compatibility Ideographs (0x2F800 - 0x2FA1D)

        if (!isPreviousCharacter)
            CFCharacterSetUnion(smartSet.get(), CFCharacterSetGetPredefined(kCFCharacterSetPunctuation));
    }
    return smartSet.get();
}

bool isCharacterSmartReplaceExempt(char32_t c, bool isPreviousCharacter)
{
    return CFCharacterSetIsLongCharacterMember(getSmartSet(isPreviousCharacter), c);
}

}
