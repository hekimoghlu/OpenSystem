/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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

// Fuzzer for ICU PluralRules.

#include <cstring>

#include "fuzzer_utils.h"

#include "unicode/plurrule.h"
#include "unicode/locid.h"


void TestPluralRules(icu::PluralRules* pp, int32_t number, double dbl,  UErrorCode& status) {
    if (U_FAILURE(status)) {
        return;
    }
    pp->select(number);
    pp->select(dbl);
}

void TestPluralRulesWithLocale(
    const icu::Locale& locale, int32_t number, double dbl,  UPluralType type, UErrorCode& status) {
    if (U_FAILURE(status)) {
        return;
    }
    std::unique_ptr<icu::PluralRules> pp(icu::PluralRules::forLocale(locale, status));
    TestPluralRules(pp.get(), number, dbl, status);

    status = U_ZERO_ERROR;
    pp.reset(icu::PluralRules::forLocale(locale, type, status));
    TestPluralRules(pp.get(), number, dbl, status);

    type = static_cast<UPluralType>(
        static_cast<int>(type) % (static_cast<int>(UPLURAL_TYPE_COUNT)));

    status = U_ZERO_ERROR;
    pp.reset(icu::PluralRules::forLocale(locale, type, status));
    TestPluralRules(pp.get(), number, dbl, status);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    uint16_t rnd;
    int32_t number;
    double dbl;
    UPluralType  type;
    if (size > 5000) {
        size = 5000;
    }
    if (size < sizeof(rnd) + sizeof(number) + sizeof(dbl) + sizeof(type)) return 0;
    icu::StringPiece fuzzData(reinterpret_cast<const char *>(data), size);

    std::memcpy(&rnd, fuzzData.data(), sizeof(rnd));
    icu::Locale locale = GetRandomLocale(rnd);
    fuzzData.remove_prefix(sizeof(rnd));

    std::memcpy(&number, fuzzData.data(), sizeof(number));
    fuzzData.remove_prefix(sizeof(number));

    std::memcpy(&dbl, fuzzData.data(), sizeof(dbl));
    fuzzData.remove_prefix(sizeof(dbl));

    std::memcpy(&type, fuzzData.data(), sizeof(type));
    fuzzData.remove_prefix(sizeof(type));

    size_t len = fuzzData.size() / sizeof(char16_t);
    icu::UnicodeString text(false, reinterpret_cast<const char16_t*>(fuzzData.data()), len);

    UErrorCode status = U_ZERO_ERROR;
    std::unique_ptr<icu::PluralRules> pp(
        icu::PluralRules::createRules(text, status));
    TestPluralRules(pp.get(), number, dbl, status);

    status = U_ZERO_ERROR;
    TestPluralRulesWithLocale(locale, number, dbl, type, status);

    std::string str(fuzzData.data(), fuzzData.size()); // ensure null-terminate
                                                       // by std::string c_str
    icu::Locale locale2(str.c_str());

    status = U_ZERO_ERROR;
    TestPluralRulesWithLocale(locale2, number, dbl, type, status);
    return EXIT_SUCCESS;
}
