/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#include <sys/cdefs.h>
__RCSID("$FreeBSD$");

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

int
main(void)
{
	bool known_arch_type;

	known_arch_type = false;
#ifdef	__LP64__
	printf("LP64\n");
	known_arch_type = true;
#endif
#ifdef	__LP32__
	printf("LP32\n");
	known_arch_type = true;
#endif
#ifdef	__ILP32__
	printf("ILP32\n");
	known_arch_type = true;
#endif

	if (known_arch_type)
		exit(0);

	fprintf(stderr, "unknown architecture type detected\n");
	assert(0);
}
