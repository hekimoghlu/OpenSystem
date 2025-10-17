/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
#include "nasmlib.h"

const char * const nasmlib_digit_chars[2] = {
    /* Lower case version */
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "@_",

    /* Upper case version */
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "@_"
};

/*
 * Produce an unsigned integer string from a number with a specified
 * base, digits and signedness.
 */
int numstr(char *buf, size_t buflen, uint64_t n,
           int digits, unsigned int base, bool ucase)
{
    const char * const dchars = nasm_digit_chars(ucase);
    bool moredigits = digits <= 0;
    char *p;
    int len;

    if (base < 2 || base > NUMSTR_MAXBASE)
        return -1;

    if (moredigits)
        digits = -digits;

    p = buf + buflen;
    *--p = '\0';

    while (p > buf && (digits-- > 0 || (moredigits && n))) {
        *--p = dchars[n % base];
        n /= base;
    }

    len = buflen - (p - buf);   /* Including final null */
    if (p != buf)
        memmove(buf, p, len);

    return len - 1;
}
