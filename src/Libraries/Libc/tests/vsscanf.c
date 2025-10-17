/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 4, 2025.
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

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include <darwintest.h>
#include <darwintest_utils.h>

static wchar_t arg1[45] = L"Sierra";

static char s[50] = "\0";

int read_this(char *, ...);

T_DECL(test_vsscanf, "vsscanf should not modify the output string if there is a character mismatch")
{
    (void)strcpy(s,"Yosemite");
    (void)wcscpy(arg1,L"FooBarBaz");
    wprintf(L"Before vsscanf: arg1 = %S", arg1);
    (void)read_this("%l[QZxp]",arg1);

    wprintf(L"After vsscanf: arg1 = %S", arg1);
    if (wcscmp(arg1,L"FooBarBaz")) {
        T_LOG("vsscanf assigned something with %%l[] and ");
        T_LOG("input did not match.");
        T_FAIL("output string was modified");
    } else {
        T_PASS("output string is intact");
    }
}

int read_this(char *format, ...)
{
        int ret = 0;
        va_list args;

        va_start(args, format);
        ret = vsscanf(s, format, args);
        va_end(args);
        return(ret);
}
