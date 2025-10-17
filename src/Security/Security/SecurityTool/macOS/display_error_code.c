/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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
#include "display_error_code.h"
#include "security_tool.h"
#include <Security/cssmapple.h>
#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <libkern/OSByteOrder.h>

// cssmErrorString
#include <Security/SecBasePriv.h>


int display_error_code(int argc, char *const *argv)
{
	CSSM_RETURN error;
	int ix = 0;

	for (ix = 0; ix < argc; ix++)
	{
		if (strcmp("error", argv[ix])==0)
			continue;
		// set base to 0 to have it interpret radix automatically
		error = (CSSM_RETURN) strtoul(argv[ix], NULL, 0);
		printf("Error: 0x%08X %d %s\n", error, error, cssmErrorString(error));
	}

	return 1;
}
