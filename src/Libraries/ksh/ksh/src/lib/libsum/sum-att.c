/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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
#pragma prototyped

/*
 * att
 */

#define att_description	\
	"The system 5 release 4 checksum. This is the default for \bsum\b \
	when \bgetconf UNIVERSE\b is \batt\b. This is the only true sum; \
	all of the other methods are order dependent."
#define att_options	0
#define att_match	"att|sys5|s5|default"
#define att_open	long_open
#define att_init	long_init
#define att_print	long_print
#define att_data	long_data
#define att_scale	512

static int
att_block(register Sum_t* p, const void* s, size_t n)
{
	register uint32_t	c = ((Integral_t*)p)->sum;
	register unsigned char*	b = (unsigned char*)s;
	register unsigned char*	e = b + n;

	while (b < e)
		c += *b++;
	((Integral_t*)p)->sum = c;
	return 0;
}

static int
att_done(Sum_t* p)
{
	register uint32_t	c = ((Integral_t*)p)->sum;

	c = (c & 0xffff) + ((c >> 16) & 0xffff);
	c = (c & 0xffff) + (c >> 16);
	((Integral_t*)p)->sum = c & 0xffff;
	return short_done(p);
}
