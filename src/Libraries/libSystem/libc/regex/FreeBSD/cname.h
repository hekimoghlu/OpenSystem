/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
/* character-name table */
static const struct cname {
	const wchar_t *name;
	wchar_t code;
} cnames[] = {
	{L"",				(wchar_t)-1}, /* sentinel */
	{L"ACK",			L'\006'},
	{L"BEL",			L'\007'},
	{L"BS",				L'\010'},
	{L"CAN",			L'\030'},
	{L"CR",				L'\015'},
	{L"DC1",			L'\021'},
	{L"DC2",			L'\022'},
	{L"DC3",			L'\023'},
	{L"DC4",			L'\024'},
	{L"DEL",			L'\177'},
	{L"DLE",			L'\020'},
	{L"EM",				L'\031'},
	{L"ENQ",			L'\005'},
	{L"EOT",			L'\004'},
	{L"ESC",			L'\033'},
	{L"ETB",			L'\027'},
	{L"ETX",			L'\003'},
	{L"FF",				L'\014'},
	{L"FS",				L'\034'},
	{L"GS",				L'\035'},
	{L"HT",				L'\011'},
	{L"IS1",			L'\037'},
	{L"IS2",			L'\036'},
	{L"IS3",			L'\035'},
	{L"IS4",			L'\034'},
	{L"LF",				L'\012'},
	{L"NAK",			L'\025'},
	{L"NUL",			L'\0'},
	{L"RS",				L'\036'},
	{L"SI",				L'\017'},
	{L"SO",				L'\016'},
	{L"SOH",			L'\001'},
	{L"STX",			L'\002'},
	{L"SUB",			L'\032'},
	{L"SYN",			L'\026'},
	{L"US",				L'\037'},
	{L"VT",				L'\013'},
	{L"alert",			L'\007'},
	{L"ampersand",			L'&'},
	{L"apostrophe",			L'\''},
	{L"asterisk",			L'*'},
	{L"backslash",			L'\\'},
	{L"backspace",			L'\b'},
	{L"carriage-return",		L'\r'},
	{L"circumflex",			L'^'},
	{L"circumflex-accent",		L'^'},
	{L"colon",			L':'},
	{L"comma",			L','},
	{L"commercial-at",		L'@'},
	{L"dollar-sign",		L'$'},
	{L"eight",			L'8'},
	{L"equals-sign",		L'='},
	{L"exclamation-mark",		L'!'},
	{L"five",			L'5'},
	{L"form-feed",			L'\f'},
	{L"four",			L'4'},
	{L"full-stop",			L'.'},
	{L"grave-accent",		L'`'},
	{L"greater-than-sign",		L'>'},
	{L"hyphen",			L'-'},
	{L"hyphen-minus",		L'-'},
	{L"left-brace",			L'{'},
	{L"left-curly-bracket",		L'{'},
	{L"left-parenthesis",		L'('},
	{L"left-square-bracket",	L'['},
	{L"less-than-sign",		L'<'},
	{L"low-line",			L'_'},
	{L"newline",			L'\n'},
	{L"nine",			L'9'},
	{L"number-sign",		L'#'},
	{L"one",			L'1'},
	{L"percent-sign",		L'%'},
	{L"period",			L'.'},
	{L"plus-sign",			L'+'},
	{L"question-mark",		L'?'},
	{L"quotation-mark",		L'"'},
	{L"reverse-solidus",		L'\\'},
	{L"right-brace",		L'}'},
	{L"right-curly-bracket",	L'}'},
	{L"right-parenthesis",		L')'},
	{L"right-square-bracket",	L']'},
	{L"semicolon",			L';'},
	{L"seven",      		L'7'},
	{L"six",			L'6'},
	{L"slash",			L'/'},
	{L"solidus",			L'/'},
	{L"space",			L' '},
	{L"tab",			L'\t'},
	{L"three",			L'3'},
	{L"tilde",			L'~'},
	{L"two",			L'2'},
	{L"underscore",			L'_'},
	{L"vertical-line",		L'|'},
	{L"vertical-tab",		L'\v'},
	{L"zero",			L'0'},
};

#define NCNAMES		(sizeof(cnames) / sizeof(*cnames))
