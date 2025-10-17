/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#include "unicode/unistr.h"
#include <stdio.h>
#include <stdlib.h>

using namespace icu;

// Verify that a UErrorCode is successful; exit(1) if not
void check(UErrorCode& status, const char* msg) {
    if (U_FAILURE(status)) {
        printf("ERROR: %s (%s)\n", u_errorName(status), msg);
        exit(1);
    }
    // printf("Ok: %s\n", msg);
}
                                                      
// Append a hex string to the target
static UnicodeString& appendHex(uint32_t number, 
                         int8_t digits, 
                         UnicodeString& target) {
    static const UnicodeString DIGIT_STRING("0123456789ABCDEF");
    while (digits > 0) {
        target += DIGIT_STRING[(number >> ((--digits) * 4)) & 0xF];
    }
    return target;
}

// Replace nonprintable characters with unicode escapes
UnicodeString escape(const UnicodeString &source) {
    int32_t i;
    UnicodeString target;
    target += "\"";
    for (i=0; i<source.length(); ++i) {
        char16_t ch = source[i];
        if (ch < 0x09 || (ch > 0x0A && ch < 0x20) || ch > 0x7E) {
            target += "\\u";
            appendHex(ch, 4, target);
        } else {
            target += ch;
        }
    }
    target += "\"";
    return target;
}

// Print the given string to stdout
void uprintf(const UnicodeString &str) {
    char* buf = nullptr;
    int32_t len = str.length();
    // int32_t bufLen = str.extract(0, len, buf); // Preflight
    /* Preflighting seems to be broken now, so assume 1-1 conversion,
       plus some slop. */
    int32_t bufLen = len + 16;
	int32_t actualLen;
    buf = new char[bufLen + 1];
    actualLen = str.extract(0, len, buf/*, bufLen*/); // Default codepage conversion
    buf[actualLen] = 0;
    printf("%s", buf);
    delete [] buf;
}
