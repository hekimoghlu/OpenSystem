/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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

/* If this symbol is defined then ENGINE_get_cipher_engine(), the function that
 * is used by EVP to hook in cipher code and cache defaults (etc), will display
 * brief debugging summaries to stderr with the 'nid'. */
/* #define ENGINE_CIPHER_DEBUG */

static ENGINE_TABLE *cipher_table = NULL;

void ENGINE_unregister_ciphers(ENGINE *e)
	{
	engine_table_unregister(&cipher_table, e);
	}

static void engine_unregister_all_ciphers(void)
	{
	engine_table_cleanup(&cipher_table);
	}

int ENGINE_register_ciphers(ENGINE *e)
	{
	if(e->ciphers)
		{
		const int *nids;
		int num_nids = e->ciphers(e, NULL, &nids, 0);
		if(num_nids > 0)
			return engine_table_register(&cipher_table,
					engine_unregister_all_ciphers, e, nids,
					num_nids, 0);
		}
	return 1;
	}

void ENGINE_register_all_ciphers()
	{
	ENGINE *e;

	for(e=ENGINE_get_first() ; e ; e=ENGINE_get_next(e))
		ENGINE_register_ciphers(e);
	}

int ENGINE_set_default_ciphers(ENGINE *e)
	{
	if(e->ciphers)
		{
		const int *nids;
		int num_nids = e->ciphers(e, NULL, &nids, 0);
		if(num_nids > 0)
			return engine_table_register(&cipher_table,
					engine_unregister_all_ciphers, e, nids,
					num_nids, 1);
		}
	return 1;
	}

/* Exposed API function to get a functional reference from the implementation
 * table (ie. try to get a functional reference from the tabled structural
 * references) for a given cipher 'nid' */
ENGINE *ENGINE_get_cipher_engine(int nid)
	{
	return engine_table_select(&cipher_table, nid);
	}

/* Obtains a cipher implementation from an ENGINE functional reference */
const EVP_CIPHER *ENGINE_get_cipher(ENGINE *e, int nid)
	{
	const EVP_CIPHER *ret;
	ENGINE_CIPHERS_PTR fn = ENGINE_get_ciphers(e);
	if(!fn || !fn(e, &ret, NULL, nid))
		{
		ENGINEerr(ENGINE_F_ENGINE_GET_CIPHER,
				ENGINE_R_UNIMPLEMENTED_CIPHER);
		return NULL;
		}
	return ret;
	}

/* Gets the cipher callback from an ENGINE structure */
ENGINE_CIPHERS_PTR ENGINE_get_ciphers(const ENGINE *e)
	{
	return e->ciphers;
	}

/* Sets the cipher callback in an ENGINE structure */
int ENGINE_set_ciphers(ENGINE *e, ENGINE_CIPHERS_PTR f)
	{
	e->ciphers = f;
	return 1;
	}
