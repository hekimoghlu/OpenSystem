/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
#ifndef netdissect_ctype_h
#define netdissect_ctype_h

/*
 * Locale-independent macros for testing character properties and
 * stripping the 8th bit from characters.
 *
 * Byte values outside the ASCII range are considered unprintable, so
 * both ND_ASCII_ISPRINT() and ND_ASCII_ISGRAPH() return "false" for them.
 *
 * Assumed to be handed a value between 0 and 255, i.e. don't hand them
 * a char, as those might be in the range -128 to 127.
 */
#define ND_ISASCII(c)		(!((c) & 0x80))	/* value is an ASCII code point */
#define ND_ASCII_ISPRINT(c)	((c) >= 0x20 && (c) <= 0x7E)
#define ND_ASCII_ISGRAPH(c)	((c) > 0x20 && (c) <= 0x7E)
#define ND_ASCII_ISDIGIT(c)	((c) >= '0' && (c) <= '9')
#define ND_TOASCII(c)		((c) & 0x7F)

/*
 * Locale-independent macros for coverting to upper or lower case.
 *
 * Byte values outside the ASCII range are not converted.  Byte values
 * *in* the ASCII range are converted to byte values in the ASCII range;
 * in particular, 'i' is upper-cased to 'I" and 'I' is lower-cased to 'i',
 * even in Turkish locales.
 */
#define ND_ASCII_TOLOWER(c)	(((c) >= 'A' && (c) <= 'Z') ? (c) - 'A' + 'a' : (c))
#define ND_ASCII_TOUPPER(c)	(((c) >= 'a' && (c) <= 'z') ? (c) - 'a' + 'A' : (c))

#endif /* netdissect-ctype.h */

