/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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

#include <TargetConditionals.h>
#include <locale.h>
#include <stdarg.h>
#include <stdio.h>
#include <wchar.h>
#include <wctype.h>
#include <xlocale.h>

#include <darwintest.h>

#if TARGET_OS_OSX
T_DECL(locale_PR_23679075, "converts a cyrillic a to uppercase")
{
	locale_t loc = newlocale(LC_COLLATE_MASK|LC_CTYPE_MASK, "ru_RU", 0);
	T_ASSERT_NOTNULL(loc, "newlocale(LC_COLLATE_MASK|LC_CTYPE_MASK, \"ru_RU\", 0) should return a locale");

	T_ASSERT_EQ(towupper_l(0x0430, loc), 0x0410, NULL);
	freelocale(loc);
}

T_DECL(locale_PR_24165555, "swprintf with Russian chars")
{
    setlocale(LC_ALL, "ru_RU.UTF-8");

    wchar_t buffer[256];
    T_EXPECT_POSIX_SUCCESS(swprintf(buffer, 256, L"%ls", L"English: Hello World"), "English");
    T_EXPECT_POSIX_SUCCESS(swprintf(buffer, 256, L"%ls", L"Russian: Ñ€ÑƒÌÑÑÐºÐ¸Ð¹ ÑÐ·Ñ‹ÌÐº"), "Russian");

    setlocale(LC_ALL, "");
}

T_DECL(locale_PR_28774201, "return code on bad locale")
{
    T_EXPECT_NULL(newlocale(LC_COLLATE_MASK | LC_CTYPE_MASK, "foobar", NULL), NULL);
    T_EXPECT_EQ(errno, ENOENT, NULL);
}
#endif
