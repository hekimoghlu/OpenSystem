/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 12, 2022.
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

// Â© 2023 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html

// Fuzzer for ICU Normalizer2.

#include <cstring>

#include "unicode/normalizer2.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    uint16_t rnd;
    UChar32 char1, char2;
    // To avoid timeout, limit the input to 5000 bytes
    if (size > 5000) {
        size = 5000;
    }
    if (size < sizeof(rnd) + sizeof(char1) + sizeof(char2)) return 0;
    icu::StringPiece fuzzData(reinterpret_cast<const char *>(data), size);

    std::memcpy(&rnd, fuzzData.data(), sizeof(rnd));
    fuzzData.remove_prefix(sizeof(rnd));
    std::memcpy(&char1, fuzzData.data(), sizeof(char1));
    fuzzData.remove_prefix(sizeof(char1));
    std::memcpy(&char2, fuzzData.data(), sizeof(char2));
    fuzzData.remove_prefix(sizeof(char2));

    size_t len = fuzzData.size() / sizeof(char16_t);
    icu::UnicodeString text(false, reinterpret_cast<const char16_t*>(fuzzData.data()), len);

    UErrorCode status = U_ZERO_ERROR;
    const icu::Normalizer2* norm = nullptr;
    switch (rnd % 6) {
        case 0:
            norm = icu::Normalizer2::getNFCInstance(status);
            break;
        case 1:
            norm = icu::Normalizer2::getNFDInstance(status);
            break;
        case 2:
            norm = icu::Normalizer2::getNFKCInstance(status);
            break;
        case 3:
            norm = icu::Normalizer2::getNFKDInstance(status);
            break;
        case 4:
            norm = icu::Normalizer2::getNFKCCasefoldInstance(status);
            break;
        case 5:
            norm = icu::Normalizer2::getNFKCSimpleCasefoldInstance(status);
            break;
    }
    if (U_SUCCESS(status)) {
        norm->normalize(text, status);
        status = U_ZERO_ERROR;

        icu::UnicodeString out;

        norm->normalize(text, out, status);
        status = U_ZERO_ERROR;

        norm->normalizeSecondAndAppend(out, text, status);
        status = U_ZERO_ERROR;

        norm->append(out, text, status);
        status = U_ZERO_ERROR;

        norm->getDecomposition(char1, out);
        norm->getRawDecomposition(char1, out);
        norm->composePair(char1, char2);
        norm->getCombiningClass(char1);
        norm->isNormalized(text, status);
        status = U_ZERO_ERROR;

        norm->quickCheck(text, status);
        status = U_ZERO_ERROR;

        norm->hasBoundaryBefore(char1);
        norm->hasBoundaryAfter(char1);
        norm->isInert(char1);
    }

    return EXIT_SUCCESS;
}
