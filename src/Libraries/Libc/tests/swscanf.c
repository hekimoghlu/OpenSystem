/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

#include <stdio.h>
#include <wchar.h>
#include <darwintest.h>
#include <darwintest_utils.h>

T_DECL(swscanf, "input conversion")
{
    wchar_t arg [] = L"abcd efgh ik";
    wchar_t s[50];
    int ret = 0;

    (void)wcscpy(s,L"\0");
    ret = swscanf(s,L"%[Zto]",arg);
    T_ASSERT_EQ(ret, EOF, "swscanf returned %d", ret);
}

T_DECL(swscanf_53347577, "rdar://53347577")
{
	int a = 0, b = 0, n = 0;
	T_EXPECT_EQ_INT(swscanf(L"23 19", L"%d %d%n", &a, &b, &n), 2, NULL);
	T_EXPECT_EQ_INT(a, 23, NULL);
	T_EXPECT_EQ_INT(b, 19, NULL);
	T_EXPECT_EQ_INT(n, 5, NULL);
}
