/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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

/* If this symbol is defined then ENGINE_get_digest_engine(), the function that
 * is used by EVP to hook in digest code and cache defaults (etc), will display
 * brief debugging summaries to stderr with the 'nid'. */
/* #define ENGINE_DIGEST_DEBUG */

static ENGINE_TABLE *digest_table = NULL;

void ENGINE_unregister_digests(ENGINE *e)
	{
	engine_table_unregister(&digest_table, e);
	}

static void engine_unregister_all_digests(void)
	{
	engine_table_cleanup(&digest_table);
	}

int ENGINE_register_digests(ENGINE *e)
	{
	if(e->digests)
		{
		const int *nids;
		int num_nids = e->digests(e, NULL, &nids, 0);
		if(num_nids > 0)
			return engine_table_register(&digest_table,
					engine_unregister_all_digests, e, nids,
					num_nids, 0);
		}
	return 1;
	}

void ENGINE_register_all_digests()
	{
	ENGINE *e;

	for(e=ENGINE_get_first() ; e ; e=ENGINE_get_next(e))
		ENGINE_register_digests(e);
	}

int ENGINE_set_default_digests(ENGINE *e)
	{
	if(e->digests)
		{
		const int *nids;
		int num_nids = e->digests(e, NULL, &nids, 0);
		if(num_nids > 0)
			return engine_table_register(&digest_table,
					engine_unregister_all_digests, e, nids,
					num_nids, 1);
		}
	return 1;
	}

/* Exposed API function to get a functional reference from the implementation
 * table (ie. try to get a functional reference from the tabled structural
 * references) for a given digest 'nid' */
ENGINE *ENGINE_get_digest_engine(int nid)
	{
	return engine_table_select(&digest_table, nid);
	}

/* Obtains a digest implementation from an ENGINE functional reference */
const EVP_MD *ENGINE_get_digest(ENGINE *e, int nid)
	{
	const EVP_MD *ret;
	ENGINE_DIGESTS_PTR fn = ENGINE_get_digests(e);
	if(!fn || !fn(e, &ret, NULL, nid))
		{
		ENGINEerr(ENGINE_F_ENGINE_GET_DIGEST,
				ENGINE_R_UNIMPLEMENTED_DIGEST);
		return NULL;
		}
	return ret;
	}

/* Gets the digest callback from an ENGINE structure */
ENGINE_DIGESTS_PTR ENGINE_get_digests(const ENGINE *e)
	{
	return e->digests;
	}

/* Sets the digest callback in an ENGINE structure */
int ENGINE_set_digests(ENGINE *e, ENGINE_DIGESTS_PTR f)
	{
	e->digests = f;
	return 1;
	}
