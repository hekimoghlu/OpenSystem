/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
/* character-class table */
static struct cclass {
	const char *name;
	const char *chars;
	const char *multis;
} cclasses[] = {
	{ "alnum",	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\
0123456789",				""} ,
	{ "alpha",	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
					""} ,
	{ "blank",	" \t",		""} ,
	{ "cntrl",	"\007\b\t\n\v\f\r\1\2\3\4\5\6\16\17\20\21\22\23\24\
\25\26\27\30\31\32\33\34\35\36\37\177",	""} ,
	{ "digit",	"0123456789",	""} ,
	{ "graph",	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\
0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
					""} ,
	{ "lower",	"abcdefghijklmnopqrstuvwxyz",
					""} ,
	{ "print",	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\
0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ",
					""} ,
	{ "punct",	"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
					""} ,
	{ "space",	"\t\n\v\f\r ",	""} ,
	{ "upper",	"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
					""} ,
	{ "xdigit",	"0123456789ABCDEFabcdef",
					""} ,
	{ NULL,		0,		"" }
};
