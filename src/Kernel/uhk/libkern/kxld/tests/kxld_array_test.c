/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <strings.h>

#include <mach/mach_init.h>

#include "kxld_array.h"
#include "kxld_test.h"
#include "kxld_util.h"

#define kNumStorageTestItems (u_int) (4 * PAGE_SIZE / sizeof(u_int))

int
main(int argc __unused, char *argv[] __unused)
{
	kern_return_t rval = KERN_FAILURE;
	KXLDArray array;
	u_int *item = 0;
	u_int test_num = 0;
	u_int idx = 0;
	u_int titems = 0;
	u_int storageTestItems[kNumStorageTestItems];
	u_int i = 0;

	bzero(&array, sizeof(array));

	kxld_set_logging_callback(kxld_test_log);
	kxld_set_logging_callback_data("kxld_array_test", NULL);

	kxld_log(0, 0, "%d: Initialize", ++test_num);

	titems = PAGE_SIZE / sizeof(u_int);
	rval = kxld_array_init(&array, sizeof(u_int), titems);
	assert(rval == KERN_SUCCESS);
	assert(array.nitems == titems);

	kxld_log(0, 0, "%d: Get item", ++test_num);
	idx = 0;
	item = kxld_array_get_item(&array, idx);
	assert(item);
	assert(item == kxld_array_get_slot(&array, idx));

	idx = titems - 1;
	item = kxld_array_get_item(&array, idx);
	assert(item);
	assert(item == kxld_array_get_slot(&array, idx));

	idx = titems;
	item = kxld_array_get_item(&array, idx);
	assert(!item);
	/* We allocated the max number of items that could be stored in a page,
	 * so get_slot() and get_item() are equivalent.
	 */
	assert(item == kxld_array_get_slot(&array, idx));

	kxld_log(0, 0, "%d: Resize", ++test_num);

	titems = 2 * PAGE_SIZE / sizeof(u_int) + 100;
	rval = kxld_array_resize(&array, titems);
	assert(rval == KERN_SUCCESS);
	assert(array.nitems == titems);

	kxld_log(0, 0, "%d: Get more items", ++test_num);
	idx = 0;
	item = kxld_array_get_item(&array, idx);
	assert(item);
	assert(item == kxld_array_get_slot(&array, idx));

	idx = titems - 1;
	item = kxld_array_get_item(&array, idx);
	assert(item);
	assert(item == kxld_array_get_slot(&array, idx));

	idx = titems;
	item = kxld_array_get_item(&array, idx);
	assert(!item);
	/* We allocated fewer items than could fit in a page, so get_slot() will
	 * return items even when get_item() does not.  See below for details.
	 */
	assert(item != kxld_array_get_slot(&array, idx));

	kxld_log(0, 0, "%d: Clear and attempt to get an item", ++test_num);
	(void) kxld_array_clear(&array);
	item = kxld_array_get_item(&array, 0);
	assert(!item);

	kxld_log(0, 0, "%d: Get slot", ++test_num);
	/* The array allocates its internal storage in pages. Because get_slot()
	 * fetches items based on the allocated size, not the logical size, we
	 * calculate the max items get_slot() can retrieve based on page size.
	 */
	titems = (u_int) (round_page(titems * sizeof(u_int)) / sizeof(u_int));
	assert(!item);
	item = kxld_array_get_slot(&array, 0);
	assert(item);
	item = kxld_array_get_slot(&array, titems - 1);
	assert(item);
	item = kxld_array_get_slot(&array, titems);
	assert(!item);

	kxld_log(0, 0, "%d: Reinitialize", ++test_num);

	titems = kNumStorageTestItems;
	rval = kxld_array_init(&array, sizeof(u_int), titems);
	assert(rval == KERN_SUCCESS);
	assert(array.nitems == titems);

	kxld_log(0, 0, "%d: Storage test - %d insertions and finds",
	    ++test_num, kNumStorageTestItems);
	for (i = 0; i < titems; ++i) {
		item = kxld_array_get_item(&array, i);
		assert(item);

		*item = (u_int) (random() % UINT_MAX);
		storageTestItems[i] = *item;
	}

	for (i = 0; i < titems; ++i) {
		item = kxld_array_get_item(&array, i);
		assert(item);
		assert(*item == storageTestItems[i]);
	}

	(void) kxld_array_deinit(&array);

	kxld_log(0, 0, " ");
	kxld_log(0, 0, "All tests passed!  Now check for memory leaks...");

	kxld_print_memory_report();

	return 0;
}
