/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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
#include "hx_locl.h"

/*
 * Should use two hash/tree certificates intead of a array.  Criteria
 * should be subject and subjectKeyIdentifier since those two are
 * commonly seached on in CMS and path building.
 */

struct mem_data {
    char *name;
    struct {
	unsigned long len;
	hx509_cert *val;
    } certs;
    hx509_private_key *keys;
};

static int
mem_init(hx509_context context,
	 hx509_certs certs, void **data, int flags,
	 const char *residue, hx509_lock lock)
{
    struct mem_data *mem;
    mem = calloc(1, sizeof(*mem));
    if (mem == NULL)
	return ENOMEM;
    if (residue == NULL || residue[0] == '\0')
	residue = "anonymous";
    mem->name = strdup(residue);
    if (mem->name == NULL) {
	free(mem);
	return ENOMEM;
    }
    *data = mem;
    return 0;
}

static int
mem_free(hx509_certs certs, void *data)
{
    struct mem_data *mem = data;
    unsigned long i;

    for (i = 0; i < mem->certs.len; i++)
	hx509_cert_free(mem->certs.val[i]);
    free(mem->certs.val);
    for (i = 0; mem->keys && mem->keys[i]; i++)
	hx509_private_key_free(&mem->keys[i]);
    free(mem->keys);
    free(mem->name);
    free(mem);

    return 0;
}

static int
mem_add(hx509_context context, hx509_certs certs, void *data, hx509_cert c)
{
    struct mem_data *mem = data;
    hx509_cert *val;

    val = realloc(mem->certs.val,
		  (mem->certs.len + 1) * sizeof(mem->certs.val[0]));
    if (val == NULL)
	return ENOMEM;

    mem->certs.val = val;
    mem->certs.val[mem->certs.len] = hx509_cert_ref(c);
    mem->certs.len++;

    return 0;
}

static int
mem_iter_start(hx509_context context,
	       hx509_certs certs,
	       void *data,
	       void **cursor)
{
    unsigned long *iter = malloc(sizeof(*iter));

    if (iter == NULL)
	return ENOMEM;

    *iter = 0;
    *cursor = iter;

    return 0;
}

static int
mem_iter(hx509_context contexst,
	 hx509_certs certs,
	 void *data,
	 void *cursor,
	 hx509_cert *cert)
{
    unsigned long *iter = cursor;
    struct mem_data *mem = data;

    if (*iter >= mem->certs.len) {
	*cert = NULL;
	return 0;
    }

    *cert = hx509_cert_ref(mem->certs.val[*iter]);
    (*iter)++;
    return 0;
}

static int
mem_iter_end(hx509_context context,
	     hx509_certs certs,
	     void *data,
	     void *cursor)
{
    free(cursor);
    return 0;
}

static int
mem_getkeys(hx509_context context,
	     hx509_certs certs,
	     void *data,
	     hx509_private_key **keys)
{
    struct mem_data *mem = data;
    int i;

    for (i = 0; mem->keys && mem->keys[i]; i++)
	;
    *keys = calloc(i + 1, sizeof(**keys));
    for (i = 0; mem->keys && mem->keys[i]; i++) {
	(*keys)[i] = _hx509_private_key_ref(mem->keys[i]);
	if ((*keys)[i] == NULL) {
	    while (--i >= 0)
		hx509_private_key_free(&(*keys)[i]);
	    hx509_set_error_string(context, 0, ENOMEM, "out of memory");
	    return ENOMEM;
	}
    }
    (*keys)[i] = NULL;
    return 0;
}

static int
mem_addkey(hx509_context context,
	   hx509_certs certs,
	   void *data,
	   hx509_private_key key)
{
    struct mem_data *mem = data;
    void *ptr;
    int i;

    for (i = 0; mem->keys && mem->keys[i]; i++)
	;
    ptr = realloc(mem->keys, (i + 2) * sizeof(*mem->keys));
    if (ptr == NULL) {
	hx509_set_error_string(context, 0, ENOMEM, "out of memory");
	return ENOMEM;
    }
    mem->keys = ptr;
    mem->keys[i] = _hx509_private_key_ref(key);
    mem->keys[i + 1] = NULL;
    return 0;
}


static struct hx509_keyset_ops keyset_mem = {
    "MEMORY",
    0,
    mem_init,
    NULL,
    mem_free,
    mem_add,
    NULL,
    mem_iter_start,
    mem_iter,
    mem_iter_end,
    NULL,
    mem_getkeys,
    mem_addkey
};

void
_hx509_ks_mem_register(hx509_context context)
{
    _hx509_ks_register(context, &keyset_mem);
}
