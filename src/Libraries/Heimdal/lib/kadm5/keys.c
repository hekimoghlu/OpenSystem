/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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
#include "kadm5_locl.h"

RCSID("$Id$");

/*
 * free all the memory used by (len, keys)
 */

void
_kadm5_free_keys (krb5_context context,
		  int len, Key *keys)
{
    hdb_free_keys(context, len, keys);
}

/*
 * null-ify `len', `keys'
 */

void
_kadm5_init_keys (Key *keys, int len)
{
    int i;

    for (i = 0; i < len; ++i) {
	keys[i].mkvno               = NULL;
	keys[i].salt                = NULL;
	keys[i].key.keyvalue.length = 0;
	keys[i].key.keyvalue.data   = NULL;
    }
}


/*
 * return 1 if any key in `keys1, len1' exists in `keys2, len2'
 */
static int
_kadm5_exists_keys(Key *keys1, unsigned len1, Key *keys2, unsigned len2)
{
    unsigned i, j;

    for (i = 0; i < len1; ++i) {
	int optimize = 0;

	for (j = 0; j < len2; j++) {
	    if ((keys1[i].salt != NULL && keys2[j].salt == NULL)
		|| (keys1[i].salt == NULL && keys2[j].salt != NULL))
		continue;

	    if (keys1[i].salt != NULL) {
		if (keys1[i].salt->type != keys2[j].salt->type)
		    continue;
		if (keys1[i].salt->salt.length != keys2[j].salt->salt.length)
		    continue;
		if (memcmp (keys1[i].salt->salt.data, keys2[j].salt->salt.data,
			    keys1[i].salt->salt.length) != 0)
		    continue;
	    }
	    if (keys1[i].key.keytype != keys2[j].key.keytype)
		continue;
	    optimize = 1;
	    if (keys1[i].key.keyvalue.length != keys2[j].key.keyvalue.length)
		continue;
	    if (memcmp (keys1[i].key.keyvalue.data, keys2[j].key.keyvalue.data,
			keys1[i].key.keyvalue.length) != 0)
		continue;

	    return 1;
	}

	/*
	 * Optimization: no need to check all of keys1[] if one there
	 * was one key in keys2[] with matching enctype and salt but not
	 * matching key.  Assumption: all keys in keys1[] and keys2[]
	 * are output by string2key.
	 */
	if (optimize)
	    return 0;
    }
    return 0;
}

/*
 * return 1 if any key in `keys1, len1' exists in hist_keys
 */
int
_kadm5_exists_keys_hist(Key *keys1, unsigned len1, HDB_Ext_KeySet *hist_keys)
{
    unsigned n;

    for (n = 0; n < hist_keys->len; n++) {
	if (_kadm5_exists_keys(keys1, len1,
			       hist_keys->val[n].keys.val,
			       hist_keys->val[n].keys.len))
	    return 1;
    }

    return 0;
}
