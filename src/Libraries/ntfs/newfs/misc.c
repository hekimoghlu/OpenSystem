/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "types.h"
#include "misc.h"
#include "logging.h"

/**
 * ntfs_calloc
 * 
 * Return a pointer to the allocated memory or NULL if the request fails.
 */
void *ntfs_calloc(size_t size)
{
	void *p;
	
	p = calloc(1, size);
	if (!p)
		ntfs_log_perror("Failed to calloc %lld bytes", (long long)size);
	return p;
}

void *ntfs_malloc(size_t size)
{
	void *p;
	
	p = malloc(size);
	if (!p)
		ntfs_log_perror("Failed to malloc %lld bytes", (long long)size);
	return p;
}
