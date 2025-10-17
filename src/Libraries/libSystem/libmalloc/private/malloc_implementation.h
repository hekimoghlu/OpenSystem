/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
/* Private interfaces between libsystem_malloc, libSystem, and MallocStackLogging */

#include <malloc/malloc.h>

#ifndef _MALLOC_IMPLEMENTATION_H_
#define _MALLOC_IMPLEMENTATION_H_


/*********	Libsystem initializers ************/

struct _malloc_late_init {
	unsigned long version;
	/* The following functions are included in version 1 of this structure */
	void * (*dlopen) (const char *path, int mode);
	void * (*dlsym) (void *handle, const char *symbol);
	bool internal_diagnostics;  /* os_variant_has_internal_diagnostics() */
};

void __malloc_init(const char *apple[]);
void __malloc_late_init(const struct _malloc_late_init *);



/*
 * Definitions intended for the malloc stack logging library only.
 * This is SPI that is *not* intended for use elsewhere. It will change
 * and will eventually be removed, without prior warning.
 */
#if MALLOC_ENABLE_MSL_LITE_SPI

typedef struct szone_s szone_t;

typedef struct _malloc_msl_lite_hooks_s {
	szone_t *(*create_and_insert_msl_lite_zone)(const char *name,
											void *mallocp, void *callocp,
											void *vallocp, void *reallocp, void *batch_mallocp,
											void *batch_freep, void *memalignp, void *freep,
											void *free_definite_sizep, void *sizep);
	malloc_zone_t *(*helper_zone)(szone_t *zone);
	size_t (*szone_size)(szone_t *szone, const void *ptr);
	void *(*szone_malloc)(szone_t *szone, size_t size);
	void *(*szone_malloc_should_clear)(szone_t *szone, size_t size,
									   boolean_t cleared_requested);
	void (*szone_free)(szone_t *szone, void *ptr);
	void *(*szone_realloc)(szone_t *szone, void *ptr, size_t new_size);
	void *(*szone_valloc)(szone_t *szone, size_t size);
	void *(*szone_memalign)(szone_t *szone, size_t alignment, size_t size);
	unsigned (*szone_batch_malloc)(szone_t *szone, size_t size, void **results,
								   unsigned count);
	void (*szone_batch_free)(szone_t *szone, void **to_be_freed, unsigned count);
	boolean_t (*has_default_zone0)(void);
	
	size_t (*calloc_get_size)(size_t num_items, size_t size, size_t extra_size,
							  size_t *total_size);

	size_t (*szone_good_size)(szone_t *szone, size_t size);
	malloc_zone_t *(*basic_zone)(szone_t *zone);
} _malloc_msl_lite_hooks_t;

#endif // MALLOC_ENABLE_MSL_LITE_SPI

#endif

