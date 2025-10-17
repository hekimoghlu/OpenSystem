/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
#include "unaccent.h"

using icu::Replaceable;
using icu::Transliterator;
using icu::UnicodeString;

UOBJECT_DEFINE_RTTI_IMPLEMENTATION(UnaccentTransliterator)

/**
 * Constructor
 */
UnaccentTransliterator::UnaccentTransliterator() :
    normalizer("", UNORM_NFD),
    Transliterator("Unaccent", nullptr) {
}

/**
 * Destructor
 */
UnaccentTransliterator::~UnaccentTransliterator() {
}

/**
 * Remove accents from a character using Normalizer.
 */
char16_t UnaccentTransliterator::unaccent(char16_t c) const {
    UnicodeString str(c);
    UErrorCode status = U_ZERO_ERROR;
    UnaccentTransliterator* t = const_cast<UnaccentTransliterator*>(this);

    t->normalizer.setText(str, status);
    if (U_FAILURE(status)) {
        return c;
    }
    return static_cast<char16_t>(t->normalizer.next());
}

/**
 * Implement Transliterator API
 */
void UnaccentTransliterator::handleTransliterate(Replaceable& text,
                                                 UTransPosition& index,
                                                 UBool incremental) const {
    UnicodeString str("a");
    while (index.start < index.limit) {
        char16_t c = text.charAt(index.start);
        char16_t d = unaccent(c);
        if (c != d) {
            str.setCharAt(0, d);
            text.handleReplaceBetween(index.start, index.start+1, str);
        }
        index.start++;
    }
}
