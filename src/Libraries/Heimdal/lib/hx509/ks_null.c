/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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

#ifdef HEIM_KS_NULL

static int
null_init(hx509_context context,
	  hx509_certs certs, void **data, int flags,
	  const char *residue, hx509_lock lock)
{
    *data = NULL;
    return 0;
}

static int
null_free(hx509_certs certs, void *data)
{
    assert(data == NULL);
    return 0;
}

static int
null_iter_start(hx509_context context,
		hx509_certs certs, void *data, void **cursor)
{
    *cursor = NULL;
    return 0;
}

static int
null_iter(hx509_context context,
	  hx509_certs certs, void *data, void *iter, hx509_cert *cert)
{
    *cert = NULL;
    return ENOENT;
}

static int
null_iter_end(hx509_context context,
	      hx509_certs certs,
	      void *data,
	      void *cursor)
{
    assert(cursor == NULL);
    return 0;
}


struct hx509_keyset_ops keyset_null = {
    "NULL",
    0,
    null_init,
    NULL,
    null_free,
    NULL,
    NULL,
    null_iter_start,
    null_iter,
    null_iter_end
};

#endif

void
_hx509_ks_null_register(hx509_context context)
{
#ifdef HEIM_KS_NULL
    _hx509_ks_register(context, &keyset_null);
#endif
}
