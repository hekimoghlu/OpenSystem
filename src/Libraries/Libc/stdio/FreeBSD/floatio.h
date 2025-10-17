/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
/*
 * Floating point scanf/printf (input/output) definitions.
 */

/*
 * MAXEXPDIG is the maximum number of decimal digits needed to store a
 * floating point exponent in the largest supported format.  It should
 * be ceil(log10(LDBL_MAX_10_EXP)) or, if hexadecimal floating point
 * conversions are supported, ceil(log10(LDBL_MAX_EXP)).  But since it
 * is presently never greater than 5 in practice, we fudge it.
 */
#define	MAXEXPDIG	6
#if LDBL_MAX_EXP > 999999
#error "floating point buffers too small"
#endif

char *__hdtoa(double, const char *, int, int *, int *, char **);
char *__hldtoa(long double, const char *, int, int *, int *, char **);
char *__ldtoa(long double *, int, int, int *, int *, char **);
