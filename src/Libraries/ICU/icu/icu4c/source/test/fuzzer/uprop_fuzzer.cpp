/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 13, 2022.
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

// Fuzzer for ICU Unicode Property.

#include <cstring>

#include "fuzzer_utils.h"

#include "unicode/uchar.h"
#include "unicode/locid.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    UProperty prop;
    UChar32 c32;

    if (size < sizeof(prop) + sizeof(c32)) return 0;

    icu::StringPiece fuzzData(reinterpret_cast<const char *>(data), size);

    std::memcpy(&prop, fuzzData.data(), sizeof(prop));
    fuzzData.remove_prefix(sizeof(prop));

    std::memcpy(&c32, fuzzData.data(), sizeof(c32));
    fuzzData.remove_prefix(sizeof(c32));

    u_hasBinaryProperty(c32, prop);

    UErrorCode status = U_ZERO_ERROR;
    u_getBinaryPropertySet(prop, &status);

    u_getIntPropertyValue(c32, prop);
    u_getIntPropertyMinValue(prop);
    u_getIntPropertyMaxValue(prop);

    status = U_ZERO_ERROR;
    u_getIntPropertyMap(prop, &status);

    size_t unistr_size = fuzzData.length()/2;
    const UChar* p = (const UChar*)(fuzzData.data());
    u_stringHasBinaryProperty(p, unistr_size, prop);

    return 0;
}
