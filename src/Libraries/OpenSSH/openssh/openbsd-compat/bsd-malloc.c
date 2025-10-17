/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
#include "config.h"
#undef malloc
#undef calloc
#undef realloc

#include <sys/types.h>
#include <stdlib.h>

#if defined(HAVE_MALLOC) && HAVE_MALLOC == 0
void *
rpl_malloc(size_t size)
{
	if (size == 0)
		size = 1;
	return malloc(size);
}
#endif

#if defined(HAVE_CALLOC) && HAVE_CALLOC == 0
void *
rpl_calloc(size_t nmemb, size_t size)
{
	if (nmemb == 0)
		nmemb = 1;
	if (size == 0)
		size = 1;
	return calloc(nmemb, size);
}
#endif

#if defined (HAVE_REALLOC) && HAVE_REALLOC == 0
void *
rpl_realloc(void *ptr, size_t size)
{
	if (size == 0)
		size = 1;
	if (ptr == 0)
		return malloc(size);
	return realloc(ptr, size);
}
#endif
