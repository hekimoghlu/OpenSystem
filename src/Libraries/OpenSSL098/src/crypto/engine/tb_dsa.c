/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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

/* If this symbol is defined then ENGINE_get_default_DSA(), the function that is
 * used by DSA to hook in implementation code and cache defaults (etc), will
 * display brief debugging summaries to stderr with the 'nid'. */
/* #define ENGINE_DSA_DEBUG */

static ENGINE_TABLE *dsa_table = NULL;
static const int dummy_nid = 1;

void ENGINE_unregister_DSA(ENGINE *e)
	{
	engine_table_unregister(&dsa_table, e);
	}

static void engine_unregister_all_DSA(void)
	{
	engine_table_cleanup(&dsa_table);
	}

int ENGINE_register_DSA(ENGINE *e)
	{
	if(e->dsa_meth)
		return engine_table_register(&dsa_table,
				engine_unregister_all_DSA, e, &dummy_nid, 1, 0);
	return 1;
	}

void ENGINE_register_all_DSA()
	{
	ENGINE *e;

	for(e=ENGINE_get_first() ; e ; e=ENGINE_get_next(e))
		ENGINE_register_DSA(e);
	}

int ENGINE_set_default_DSA(ENGINE *e)
	{
	if(e->dsa_meth)
		return engine_table_register(&dsa_table,
				engine_unregister_all_DSA, e, &dummy_nid, 1, 1);
	return 1;
	}

/* Exposed API function to get a functional reference from the implementation
 * table (ie. try to get a functional reference from the tabled structural
 * references). */
ENGINE *ENGINE_get_default_DSA(void)
	{
	return engine_table_select(&dsa_table, dummy_nid);
	}

/* Obtains an DSA implementation from an ENGINE functional reference */
const DSA_METHOD *ENGINE_get_DSA(const ENGINE *e)
	{
	return e->dsa_meth;
	}

/* Sets an DSA implementation in an ENGINE structure */
int ENGINE_set_DSA(ENGINE *e, const DSA_METHOD *dsa_meth)
	{
	e->dsa_meth = dsa_meth;
	return 1;
	}
