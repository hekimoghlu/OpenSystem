/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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
#include <ctype.h>
#include <stdio.h>          // For stdout, stderr

#include "ClientCommon.h"

const char *GetNextLabel(const char *cstr, char label[64])
{
    char *ptr = label;
    while (*cstr && *cstr != '.')               // While we have characters in the label...
    {
        char c = *cstr++;
        if (c == '\\')                          // If escape character, check next character
        {
            if (*cstr == '\0') break;           // If this is the end of the string, then break
            c = *cstr++;
            if (isdigit(cstr[-1]) && isdigit(cstr[0]) && isdigit(cstr[1]))
            {
                int v0 = cstr[-1] - '0';                        // then interpret as three-digit decimal
                int v1 = cstr[ 0] - '0';
                int v2 = cstr[ 1] - '0';
                int val = v0 * 100 + v1 * 10 + v2;
                // If valid three-digit decimal value, use it
                // Note that although ascii nuls are possible in DNS labels
                // we're building a C string here so we have no way to represent that
                if (val == 0) val = '-';
                if (val <= 255) { c = (char)val; cstr += 2; }
            }
        }
        *ptr++ = c;
        if (ptr >= label+64) { label[63] = 0; return(NULL); }   // Illegal label more than 63 bytes
    }
    *ptr = 0;                                                   // Null-terminate label text
    if (ptr == label) return(NULL);                             // Illegal empty label
    if (*cstr) cstr++;                                          // Skip over the trailing dot (if present)
    return(cstr);
}
