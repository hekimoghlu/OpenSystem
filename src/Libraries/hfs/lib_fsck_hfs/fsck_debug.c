/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include <stdarg.h>
#include "fsck_debug.h"
#include "check.h"

void HexDump(const void *p_arg, unsigned length, int showOffsets)
{
	const u_int8_t *p = p_arg;
	unsigned i;
	char ascii[17];
	u_int8_t byte;
	
	ascii[16] = '\0';
	
	for (i=0; i<length; ++i)
	{
		if (showOffsets && (i & 0xF) == 0)
            fsck_print(ctx, LOG_TYPE_INFO, "%08X: ", i);

		byte = p[i];
        fsck_print(ctx, LOG_TYPE_INFO, "%02X ", byte);
		if (byte < 32 || byte > 126)
			ascii[i & 0xF] = '.';
		else
			ascii[i & 0xF] = byte;
		
		if ((i & 0xF) == 0xF)
		{
            fsck_print(ctx, LOG_TYPE_INFO, "  %s\n", ascii);
		}
	}
	
	if (i & 0xF)
	{
		unsigned j;
		for (j = i & 0xF; j < 16; ++j)
            fsck_print(ctx, LOG_TYPE_INFO, "   ");
		ascii[i & 0xF] = 0;
        fsck_print(ctx, LOG_TYPE_INFO, "  %s\n", ascii);
	}
}
