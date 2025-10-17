/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
 * Copyright 2012 (c), Joyent, Inc.  All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include "usdt.h"

#define	FMT	"{" \
		"  \"sizes\": [ \"first\", 2, %f ]," \
		"  \"index\": %d," \
		"  \"facts\": {" \
		"    \"odd\": \"%s\"," \
		"    \"even\": \"%s\"" \
		"  }," \
		"  \"action\": \"%s\"" \
		"}\n"

int
waiting(volatile int *a)
{
	return (*a);
}

int
main(int argc, char **argv)
{
	volatile int a = 0;
	int idx;
	double size = 250.5;

	while (waiting(&a) == 0)
		continue;

	for (idx = 0; idx < 10; idx++) {
		char *odd, *even, *json, *action;

		size *= 1.78;
		odd = idx % 2 == 1 ? "true" : "false";
		even = idx % 2 == 0 ? "true" : "false";
		action = idx == 7 ? "ignore" : "print";

		asprintf(&json, FMT, size, idx, odd, even, action);
		BUNYAN_FAKE_LOG_DEBUG(json);
		free(json);
	}

	BUNYAN_FAKE_LOG_DEBUG("{\"finished\": true}");

	return (0);
}
