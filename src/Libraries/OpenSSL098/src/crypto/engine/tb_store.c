/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#include "eng_int.h"

/* If this symbol is defined then ENGINE_get_default_STORE(), the function that is
 * used by STORE to hook in implementation code and cache defaults (etc), will
 * display brief debugging summaries to stderr with the 'nid'. */
/* #define ENGINE_STORE_DEBUG */

static ENGINE_TABLE *store_table = NULL;
static const int dummy_nid = 1;

void ENGINE_unregister_STORE(ENGINE *e)
	{
	engine_table_unregister(&store_table, e);
	}

static void engine_unregister_all_STORE(void)
	{
	engine_table_cleanup(&store_table);
	}

int ENGINE_register_STORE(ENGINE *e)
	{
	if(e->store_meth)
		return engine_table_register(&store_table,
				engine_unregister_all_STORE, e, &dummy_nid, 1, 0);
	return 1;
	}

void ENGINE_register_all_STORE()
	{
	ENGINE *e;

	for(e=ENGINE_get_first() ; e ; e=ENGINE_get_next(e))
		ENGINE_register_STORE(e);
	}

/* The following two functions are removed because they're useless. */
#if 0
int ENGINE_set_default_STORE(ENGINE *e)
	{
	if(e->store_meth)
		return engine_table_register(&store_table,
				engine_unregister_all_STORE, e, &dummy_nid, 1, 1);
	return 1;
	}
#endif

#if 0
/* Exposed API function to get a functional reference from the implementation
 * table (ie. try to get a functional reference from the tabled structural
 * references). */
ENGINE *ENGINE_get_default_STORE(void)
	{
	return engine_table_select(&store_table, dummy_nid);
	}
#endif

/* Obtains an STORE implementation from an ENGINE functional reference */
const STORE_METHOD *ENGINE_get_STORE(const ENGINE *e)
	{
	return e->store_meth;
	}

/* Sets an STORE implementation in an ENGINE structure */
int ENGINE_set_STORE(ENGINE *e, const STORE_METHOD *store_meth)
	{
	e->store_meth = store_meth;
	return 1;
	}
