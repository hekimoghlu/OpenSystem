/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "Version64.h"

namespace mach_o {


Error Version64::fromString(CString versString, Version64& vers)
{
    // Initialize default value
    vers = Version64();
    
    uint64_t a = 0;
    uint64_t b = 0;
    uint64_t c = 0;
    uint64_t d = 0;
    uint64_t e = 0;
    char* end;
    a = strtoul(versString.c_str(), &end, 10);
    if ( *end == '.' ) {
        b = strtoul(&end[1], &end, 10);
        if ( *end == '.' ) {
            c = strtoul(&end[1], &end, 10);
            if ( *end == '.' ) {
                d = strtoul(&end[1], &end, 10);
                if ( *end == '.' ) {
                    e = strtoul(&end[1], &end, 10);
                }
            }
        }
    }
    if ( (*end != '\0') || (a > 0xFFFFFF) || (b > 0x3FF) || (c > 0x3FF) || (d > 0x3FF)  || (e > 0x3FF) )
        return Error("malformed 64-bit a.b.c.d.e version number: %s", versString.c_str());
    
    vers = Version64(a, b, c, d, e);
    return Error::none();
}

static void appendDigit(char*& s, unsigned& num, unsigned place, bool& startedPrinting)
{
    if ( num >= place ) {
        unsigned dig = (num/place);
        *s++ = '0' + dig;
        num -= (dig*place);
        startedPrinting = true;
    }
    else if ( startedPrinting ) {
        *s++ = '0';
    }
}

static void appendNumber(char*& s, unsigned num)
{
    assert(num < 9999999);
    bool startedPrinting = false;
    appendDigit(s, num, 10000000, startedPrinting);
    appendDigit(s, num,  1000000, startedPrinting);
    appendDigit(s, num,   100000, startedPrinting);
    appendDigit(s, num,    10000, startedPrinting);
    appendDigit(s, num,     1000, startedPrinting);
    appendDigit(s, num,      100, startedPrinting);
    appendDigit(s, num,       10, startedPrinting);
    appendDigit(s, num,        1, startedPrinting);
    if ( !startedPrinting )
        *s++ = '0';
}

/* A.B.C.D.E packed as a24.b10.c10.d10.e10 */
const char* Version64::toString(char buffer[64]) const
{
    char* s = buffer;
    appendNumber(s, (_raw >> 40));
    *s++ = '.';
    appendNumber(s, (_raw >> 30) & 0x3FF);
    unsigned c = (_raw >> 20) & 0x3FF;
    if ( c != 0 ) {
        *s++ = '.';
        appendNumber(s, c);
    }
    unsigned d = (_raw >> 10) & 0x3FF;
    if ( d != 0 ) {
        *s++ = '.';
        appendNumber(s, d);
    }
    unsigned e = _raw & 0x3FF;
    if ( e != 0 ) {
        *s++ = '.';
        appendNumber(s, e);
    }
    *s++ = '\0';
    return buffer;
}

} // namespace mach_o
