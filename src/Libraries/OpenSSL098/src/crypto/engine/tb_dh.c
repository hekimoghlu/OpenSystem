/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
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

/* If this symbol is defined then ENGINE_get_default_DH(), the function that is
 * used by DH to hook in implementation code and cache defaults (etc), will
 * display brief debugging summaries to stderr with the 'nid'. */
/* #define ENGINE_DH_DEBUG */

static ENGINE_TABLE *dh_table = NULL;
static const int dummy_nid = 1;

void ENGINE_unregister_DH(ENGINE *e)
	{
	engine_table_unregister(&dh_table, e);
	}

static void engine_unregister_all_DH(void)
	{
	engine_table_cleanup(&dh_table);
	}

int ENGINE_register_DH(ENGINE *e)
	{
	if(e->dh_meth)
		return engine_table_register(&dh_table,
				engine_unregister_all_DH, e, &dummy_nid, 1, 0);
	return 1;
	}

void ENGINE_register_all_DH()
	{
	ENGINE *e;

	for(e=ENGINE_get_first() ; e ; e=ENGINE_get_next(e))
		ENGINE_register_DH(e);
	}

int ENGINE_set_default_DH(ENGINE *e)
	{
	if(e->dh_meth)
		return engine_table_register(&dh_table,
				engine_unregister_all_DH, e, &dummy_nid, 1, 1);
	return 1;
	}

/* Exposed API function to get a functional reference from the implementation
 * table (ie. try to get a functional reference from the tabled structural
 * references). */
ENGINE *ENGINE_get_default_DH(void)
	{
	return engine_table_select(&dh_table, dummy_nid);
	}

/* Obtains an DH implementation from an ENGINE functional reference */
const DH_METHOD *ENGINE_get_DH(const ENGINE *e)
	{
	return e->dh_meth;
	}

/* Sets an DH implementation in an ENGINE structure */
int ENGINE_set_DH(ENGINE *e, const DH_METHOD *dh_meth)
	{
	e->dh_meth = dh_meth;
	return 1;
	}
