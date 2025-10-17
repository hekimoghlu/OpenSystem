/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
__FBSDID("$FreeBSD$");

#include <stdlib.h>

void *
reallocf(void *ptr, size_t size)
{
	void *nptr;

	nptr = realloc(ptr, size);

	/*
	 * When the System V compatibility option (malloc "V" flag) is
	 * in effect, realloc(ptr, 0) frees the memory and returns NULL.
	 * So, to avoid double free, call free() only when size != 0.
	 * realloc(ptr, 0) can't fail when ptr != NULL.
	 */
	if (!nptr && ptr && size != 0)
		free(ptr);
	return (nptr);
}
