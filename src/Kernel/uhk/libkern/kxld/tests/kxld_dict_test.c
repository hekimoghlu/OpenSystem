/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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
#include <stdlib.h>
#include <strings.h>

#include "kxld_dict.h"
#include "kxld_test.h"

#define KEYLEN 40
#define STRESSNUM 10000

typedef struct {
	char * key;
	int * value;
} Stress;

int
main(int argc __unused, char *argv[] __unused)
{
	kern_return_t result = KERN_SUCCESS;
	KXLDDict dict;
	int a1 = 1, a2 = 3, i = 0, j = 0;
	void * b = NULL;
	u_int test_num = 0;
	u_long size = 0;
	Stress stress_test[STRESSNUM];

	kxld_set_logging_callback(kxld_test_log);
	kxld_set_logging_callback_data("kxld_dict_test", NULL);

	bzero(&dict, sizeof(dict));

	kxld_log(0, 0, "%d: Initialize", ++test_num);
	result = kxld_dict_init(&dict, kxld_dict_string_hash, kxld_dict_string_cmp, 10);
	assert(result == KERN_SUCCESS);
	size = kxld_dict_get_num_entries(&dict);
	assert(size == 0);

	kxld_log(0, 0, "%d: Find nonexistant key", ++test_num);
	b = kxld_dict_find(&dict, "hi");
	assert(b == NULL);

	kxld_log(0, 0, "%d: Insert and find", ++test_num);
	result = kxld_dict_insert(&dict, "hi", &a1);
	assert(result == KERN_SUCCESS);
	b = kxld_dict_find(&dict, "hi");
	assert(b && *(int*)b == a1);
	size = kxld_dict_get_num_entries(&dict);
	assert(size == 1);

	kxld_log(0, 0, "%d: Insert same key with different values", ++test_num);
	result = kxld_dict_insert(&dict, "hi", &a2);
	assert(result == KERN_SUCCESS);
	b = kxld_dict_find(&dict, "hi");
	assert(b && *(int*)b == a2);
	size = kxld_dict_get_num_entries(&dict);
	assert(size == 1);

	kxld_log(0, 0, "%d: Clear and find of nonexistant key", ++test_num);
	kxld_dict_clear(&dict);
	result = kxld_dict_init(&dict, kxld_dict_string_hash, kxld_dict_string_cmp, 10);
	assert(result == KERN_SUCCESS);
	b = kxld_dict_find(&dict, "hi");
	assert(b == NULL);
	size = kxld_dict_get_num_entries(&dict);
	assert(size == 0);

	kxld_log(0, 0, "%d: Insert multiple keys", ++test_num);
	result = kxld_dict_insert(&dict, "hi", &a1);
	assert(result == KERN_SUCCESS);
	result = kxld_dict_insert(&dict, "hello", &a2);
	assert(result == KERN_SUCCESS);
	b = kxld_dict_find(&dict, "hi");
	assert(result == KERN_SUCCESS);
	assert(b && *(int*)b == a1);
	b = kxld_dict_find(&dict, "hello");
	assert(b && *(int*)b == a2);
	size = kxld_dict_get_num_entries(&dict);
	assert(size == 2);

	kxld_log(0, 0, "%d: Remove keys", ++test_num);
	kxld_dict_remove(&dict, "hi", &b);
	assert(b && *(int*)b == a1);
	b = kxld_dict_find(&dict, "hi");
	assert(b == NULL);
	kxld_dict_remove(&dict, "hi", &b);
	assert(b == NULL);
	size = kxld_dict_get_num_entries(&dict);
	assert(size == 1);

	kxld_log(0, 0, "%d: Stress test - %d insertions and finds", ++test_num, STRESSNUM);

	kxld_dict_clear(&dict);
	result = kxld_dict_init(&dict, kxld_dict_string_hash, kxld_dict_string_cmp, 10);
	assert(result == KERN_SUCCESS);
	for (i = 0; i < STRESSNUM; ++i) {
		int * tmp_value = kxld_alloc(sizeof(int));
		char * tmp_key = kxld_alloc(sizeof(char) * (KEYLEN + 1));

		*tmp_value = i;
		for (j = 0; j < KEYLEN; ++j) {
			tmp_key[j] = (random() % 26) + 'a';
		}
		tmp_key[KEYLEN] = '\0';

		kxld_dict_insert(&dict, tmp_key, tmp_value);
		stress_test[i].key = tmp_key;
		stress_test[i].value = tmp_value;
	}

	for (i = 0; i < STRESSNUM; ++i) {
		int target_value;
		void * tmp_value;
		char * key = stress_test[i].key;

		target_value = *stress_test[i].value;
		tmp_value = kxld_dict_find(&dict, key);
		assert(target_value == *(int *)tmp_value);

		kxld_free(stress_test[i].key, sizeof(char) * (KEYLEN + 1));
		kxld_free(stress_test[i].value, sizeof(int));
	}

	kxld_log(0, 0, "%d: Destroy", ++test_num);
	kxld_dict_deinit(&dict);

	kxld_log(0, 0, "\nAll tests passed!  Now check for memory leaks...");

	kxld_print_memory_report();

	return 0;
}
