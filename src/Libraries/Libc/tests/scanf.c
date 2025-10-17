/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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
#include <darwintest.h>
#include <darwintest_utils.h>

// This can either test libkern's sscanf, or stdio.h's.
//#define TEST_LIBKERN

#if defined(TEST_LIBKERN)
static int libkern_isspace(char c);
int libkern_sscanf(const char *ibuf, const char *fmt, ...);
int libkern_vsscanf(const char *inp, char const *fmt0, va_list ap);
# define isspace(C) libkern_isspace(C)
# define sscanf(...) libkern_sscanf(__VA_ARGS__)
# define vsscanf(...) libkern_vsscanf(__VA_ARGS__)
# include "../libkern/stdio/scanf.c"
#else
# include <stdio.h>
#endif

T_DECL(scanf_empty, "empty")
{
	T_ASSERT_EQ_INT(sscanf("", ""), 0, "empty input and format");
	T_ASSERT_EQ_INT(sscanf("", "match me"), EOF, "empty input");
	T_ASSERT_EQ_INT(sscanf("lonely", ""), 0, "empty format");
}

T_DECL(scanf_percent, "percent")
{
	T_ASSERT_EQ_INT(sscanf("%", "%%"), 0, "two percent");
}

T_DECL(scanf_character, "character")
{
	char c;
	for (char i = ' '; i <= '~'; ++i) {
		char buf[] = { i, '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%c", &c), 1, "character matched");
		T_ASSERT_EQ_INT(c, i, "character value");
	}
}

T_DECL(scanf_characters, "characters")
{
	char c[] = { 'a', 'b', 'c', 'd', 'e' };
	T_ASSERT_EQ_INT(sscanf("01234", "%4c", c), 1, "characters matched");
	T_ASSERT_EQ_INT(c[0], '0', "characters value");
	T_ASSERT_EQ_INT(c[1], '1', "characters value");
	T_ASSERT_EQ_INT(c[2], '2', "characters value");
	T_ASSERT_EQ_INT(c[3], '3', "characters value");
	T_ASSERT_EQ_INT(c[4], 'e', "characters value wasn't overwritten");
}

T_DECL(scanf_string, "string")
{
	char s[] = { 'a', 'b', 'c', 'd', 'e' };
	T_ASSERT_EQ_INT(sscanf("012", "%s", s), 1, "string matched");
	T_ASSERT_EQ_STR(s, "012", "string value");
	T_ASSERT_EQ_INT(s[4], 'e', "string value wasn't overwritten");
	T_ASSERT_EQ_INT(sscanf("ABCDE", "%3s", s), 1, "string matched");
	T_ASSERT_EQ_STR(s, "ABC", "string value");
	T_ASSERT_EQ_INT(s[4], 'e', "string value wasn't overwritten");
}

T_DECL(scanf_decimal, "decimal")
{
	int num;
	for (char i = 0; i <= 9; ++i) {
		char buf[] = { i + '0', '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%d", &num), 1, "decimal matched");
		T_ASSERT_EQ_INT(num, i, "decimal value");
	}
	for (char i = 10; i <= 99; ++i) {
		char buf[] = { i / 10 + '0', i % 10 + '0', '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%d", &num), 1, "decimal matched");
		T_ASSERT_EQ_INT(num, i, "decimal value");
	}
	for (char i = 0; i <= 9; ++i) {
		char buf[] = { '-', i + '0', '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%d", &num), 1, "negative decimal matched");
		T_ASSERT_EQ_INT(num, -i, "negative decimal value");
	}
	T_ASSERT_EQ_INT(sscanf("-2147483648", "%d", &num), 1, "INT32_MIN matched");
	T_ASSERT_EQ_INT(num, INT32_MIN, "INT32_MIN value");
	T_ASSERT_EQ_INT(sscanf("2147483647", "%d", &num), 1, "INT32_MAX matched");
	T_ASSERT_EQ_INT(num, INT32_MAX, "INT32_MAX value");
}

T_DECL(scanf_integer, "integer")
{
	int num;
	T_ASSERT_EQ_INT(sscanf("0", "%i", &num), 1, "octal integer matched");
	T_ASSERT_EQ_INT(num, 0, "octal integer value");
	for (char i = 0; i <= 7; ++i) {
		char buf[] = { '0', i + '0', '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%i", &num), 1, "octal integer matched");
		T_ASSERT_EQ_INT(num, i, "octal integer value");
	}
	for (char i = 0; i <= 9; ++i) {
		char buf[] = { '0', 'x', i + '0', '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%i", &num), 1, "hex integer matched");
		T_ASSERT_EQ_INT(num, i, "hex integer value");
	}
	for (char i = 10; i <= 15; ++i) {
		char buf[] = { '0', 'x', i - 10 + 'a', '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%i", &num), 1, "hex integer matched");
		T_ASSERT_EQ_INT(num, i, "hex integer value");
	}
}

T_DECL(scanf_unsigned, "unsigned")
{
	unsigned num;
	T_ASSERT_EQ_INT(sscanf("4294967295", "%u", &num), 1, "UINT32_MAX matched");
	T_ASSERT_EQ_UINT(num, UINT32_MAX, "UINT32_MAX value");
}

T_DECL(scanf_octal, "octal")
{
	int num;
	T_ASSERT_EQ_INT(sscanf("0", "%o", &num), 1, "octal matched");
	T_ASSERT_EQ_INT(num, 0, "octal value");
	for (char i = 0; i <= 7; ++i) {
		char buf[] = { '0', i + '0', '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%o", &num), 1, "octal matched");
		T_ASSERT_EQ_INT(num, i, "octal value");
	}
}

T_DECL(scanf_hex, "hex")
{
	int num;
	for (char i = 0; i <= 9; ++i) {
		char buf[] = { '0', 'x', i + '0', '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%x", &num), 1, "hex matched");
		T_ASSERT_EQ_INT(num, i, "hex value");
	}
	for (char i = 10; i <= 15; ++i) {
		char buf[] = { '0', 'x', i - 10 + 'a', '\0' };
		T_ASSERT_EQ_INT(sscanf(buf, "%x", &num), 1, "hex matched");
		T_ASSERT_EQ_INT(num, i, "hex value");
	}
}

T_DECL(scanf_read, "read")
{
	int val, num;
	T_ASSERT_EQ_INT(sscanf("", "%n", &num), 0, "read matched");
	T_ASSERT_EQ_INT(num, 0, "read count");
	T_ASSERT_EQ_INT(sscanf("a", "a%n", &num), 0, "read matched");
	T_ASSERT_EQ_INT(num, 1, "read count");
	T_ASSERT_EQ_INT(sscanf("ab", "a%nb", &num), 0, "read matched");
	T_ASSERT_EQ_INT(num, 1, "read count");
	T_ASSERT_EQ_INT(sscanf("1234567", "%i%n", &val, &num), 1, "read matched");
	T_ASSERT_EQ_INT(val, 1234567, "read value");
	T_ASSERT_EQ_INT(num, 7, "read count");
}

T_DECL(scanf_pointer, "pointer")
{
	void *ptr;
	if (sizeof(void*) == 4) {
		T_ASSERT_EQ_INT(sscanf("0xdeadbeef", "%p", &ptr), 1, "pointer matched");
		T_ASSERT_EQ_PTR(ptr, (void*)0xdeadbeef, "pointer value");
	} else {
		T_ASSERT_EQ_INT(sscanf("0xdeadbeefc0defefe", "%p", &ptr), 1, "pointer matched");
		T_ASSERT_EQ_PTR(ptr, (void*)0xdeadbeefc0defefe, "pointer value");
	}
}
