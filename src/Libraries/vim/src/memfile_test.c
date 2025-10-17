/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
 * memfile_test.c: Unittests for memfile.c
 * Mostly by Ivan Krasilnikov.
 */

#undef NDEBUG
#include <assert.h>

// Must include main.c because it contains much more than just main()
#define NO_VIM_MAIN
#include "main.c"

// This file has to be included because the tested functions are static
#include "memfile.c"

#define index_to_key(i) ((i) ^ 15167)
#define TEST_COUNT 50000

/*
 * Test mf_hash_*() functions.
 */
    static void
test_mf_hash(void)
{
    mf_hashtab_T   ht;
    mf_hashitem_T  *item;
    blocknr_T      key;
    long_u	   i;
    long_u	   num_buckets;

    mf_hash_init(&ht);

    // insert some items and check invariants
    for (i = 0; i < TEST_COUNT; i++)
    {
	assert(ht.mht_count == i);

	// check that number of buckets is a power of 2
	num_buckets = ht.mht_mask + 1;
	assert(num_buckets > 0 && (num_buckets & (num_buckets - 1)) == 0);

	// check load factor
	assert(ht.mht_count <= (num_buckets << MHT_LOG_LOAD_FACTOR));

	if (i < (MHT_INIT_SIZE << MHT_LOG_LOAD_FACTOR))
	{
	    // first expansion shouldn't have occurred yet
	    assert(num_buckets == MHT_INIT_SIZE);
	    assert(ht.mht_buckets == ht.mht_small_buckets);
	}
	else
	{
	    assert(num_buckets > MHT_INIT_SIZE);
	    assert(ht.mht_buckets != ht.mht_small_buckets);
	}

	key = index_to_key(i);
	assert(mf_hash_find(&ht, key) == NULL);

	// allocate and add new item
	item = LALLOC_CLEAR_ONE(mf_hashitem_T);
	assert(item != NULL);
	item->mhi_key = key;
	mf_hash_add_item(&ht, item);

	assert(mf_hash_find(&ht, key) == item);

	if (ht.mht_mask + 1 != num_buckets)
	{
	    // hash table was expanded
	    assert(ht.mht_mask + 1 == num_buckets * MHT_GROWTH_FACTOR);
	    assert(i + 1 == (num_buckets << MHT_LOG_LOAD_FACTOR));
	}
    }

    // check presence of inserted items
    for (i = 0; i < TEST_COUNT; i++)
    {
	key = index_to_key(i);
	item = mf_hash_find(&ht, key);
	assert(item != NULL);
	assert(item->mhi_key == key);
    }

    // delete some items
    for (i = 0; i < TEST_COUNT; i++)
    {
	if (i % 100 < 70)
	{
	    key = index_to_key(i);
	    item = mf_hash_find(&ht, key);
	    assert(item != NULL);
	    assert(item->mhi_key == key);

	    mf_hash_rem_item(&ht, item);
	    assert(mf_hash_find(&ht, key) == NULL);

	    mf_hash_add_item(&ht, item);
	    assert(mf_hash_find(&ht, key) == item);

	    mf_hash_rem_item(&ht, item);
	    assert(mf_hash_find(&ht, key) == NULL);

	    vim_free(item);
	}
    }

    // check again
    for (i = 0; i < TEST_COUNT; i++)
    {
	key = index_to_key(i);
	item = mf_hash_find(&ht, key);

	if (i % 100 < 70)
	{
	    assert(item == NULL);
	}
	else
	{
	    assert(item != NULL);
	    assert(item->mhi_key == key);
	}
    }

    // free hash table and all remaining items
    mf_hash_free_all(&ht);
}

    int
main(void)
{
    test_mf_hash();
    return 0;
}
