/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#include <charconv>

#include "Version32.h"

namespace mach_o {


Error Version32::fromString(std::string_view versString, Version32& vers,
                            void (^ _Nullable truncationHandler)(void))
{
    // Initialize default value
    vers = Version32();

    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    const char* endPtr = versString.data() + versString.size();
    auto res = std::from_chars(versString.data(), endPtr, x);
    if ( res.ec == std::errc{} && *res.ptr == '.' ) {
        res = std::from_chars(res.ptr + 1, endPtr, y);
        if ( res.ec == std::errc{} && *res.ptr == '.' )
            res = std::from_chars(res.ptr + 1, endPtr, z);
    }
    bool valueOverflow = (x > 0xffff) || (y > 0xff) || (z > 0xff);
    if ( valueOverflow && truncationHandler ) {
        truncationHandler();
        x = std::min(x, 0xFFFFU);
        y = std::min(y, 0xFFU);
        z = std::min(z, 0xFFU);
        valueOverflow = false;
    }
    if ( res.ptr == nullptr || res.ptr != endPtr || valueOverflow ) {
        if ( valueOverflow || (truncationHandler == nullptr) || (res.ptr == versString.data()) ) {
            char errVersString[versString.size() + 1];
            memcpy(errVersString, versString.data(), versString.size());
            errVersString[versString.size()] = 0;
            return Error("malformed version number '%s' cannot fit in 32-bit xxxx.yy.zz", (const char*)errVersString);
        }
    }

    vers = Version32(x, y, z);
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
    assert(num < 99999);
    bool startedPrinting = false;
    appendDigit(s, num, 10000, startedPrinting);
    appendDigit(s, num,  1000, startedPrinting);
    appendDigit(s, num,   100, startedPrinting);
    appendDigit(s, num,    10, startedPrinting);
    appendDigit(s, num,     1, startedPrinting);
    if ( !startedPrinting )
        *s++ = '0';
}


const char* Version32::toString(char buffer[32]) const
{
    // sprintf(versionString, "%d.%d.%d", (_raw >> 16), ((_raw >> 8) & 0xFF), (_raw & 0xFF));
    char* s = buffer;
    appendNumber(s, (_raw >> 16));
    *s++ = '.';
    appendNumber(s, (_raw >> 8) & 0xFF);
    unsigned micro = (_raw & 0xFF);
    if ( micro != 0 ) {
        *s++ = '.';
        appendNumber(s, micro);
    }
    *s++ = '\0';
    return buffer;
}



} // namespace mach_o





