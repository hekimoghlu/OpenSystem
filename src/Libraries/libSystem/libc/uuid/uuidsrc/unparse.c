/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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

#include "uuidP.h"

static const char *fmt_lower = 
	"0123456789abcdef";

static const char *fmt_upper = 
	"0123456789ABCDEF";

#ifdef UUID_UNPARSE_DEFAULT_UPPER
#define FMT_DEFAULT fmt_upper
#else
#define FMT_DEFAULT fmt_lower
#endif

static void uuid_unparse_x(const uuid_t uu, char *out, const char *fmt)
{
	const uint8_t *uuid_array = (const uint8_t *)uu;
	int uuid_index;
	
	for ( uuid_index = 0; uuid_index < sizeof(uuid_t); ++uuid_index ) {
		// insert '-' after the 4th, 6th, 8th, and 10th uuid byte
		switch (uuid_index) {
		case 4:
		case 6:
		case 8:
		case 10:
			*out++ = '-';
			break;
		}
		// insert uuid byte as two hex characters
		*out++ = fmt[*uuid_array >> 4];
		*out++ = fmt[*uuid_array++ & 0xF];
	}
	*out = 0;
}

void uuid_unparse_lower(const uuid_t uu, char *out)
{
	uuid_unparse_x(uu, out,	fmt_lower);
}

void uuid_unparse_upper(const uuid_t uu, char *out)
{
	uuid_unparse_x(uu, out,	fmt_upper);
}

void uuid_unparse(const uuid_t uu, char *out)
{
	uuid_unparse_x(uu, out, FMT_DEFAULT);
}
