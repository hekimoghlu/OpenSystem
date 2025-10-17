/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#include <config.h>

#include "roken.h"

/*
 * return a malloced copy of `h'
 */

ROKEN_LIB_FUNCTION struct hostent * ROKEN_LIB_CALL
copyhostent (const struct hostent *h)
{
    struct hostent *res;
    char **p;
    int i, n;

    res = malloc (sizeof (*res));
    if (res == NULL)
	return NULL;
    res->h_name      = NULL;
    res->h_aliases   = NULL;
    res->h_addrtype  = h->h_addrtype;
    res->h_length    = h->h_length;
    res->h_addr_list = NULL;
    res->h_name = strdup (h->h_name);
    if (res->h_name == NULL) {
	freehostent (res);
	return NULL;
    }
    for (n = 0, p = h->h_aliases; *p != NULL; ++p)
	++n;
    res->h_aliases = malloc ((n + 1) * sizeof(*res->h_aliases));
    if (res->h_aliases == NULL) {
	freehostent (res);
	return NULL;
    }
    for (i = 0; i < n + 1; ++i)
	res->h_aliases[i] = NULL;
    for (i = 0; i < n; ++i) {
	res->h_aliases[i] = strdup (h->h_aliases[i]);
	if (res->h_aliases[i] == NULL) {
	    freehostent (res);
	    return NULL;
	}
    }

    for (n = 0, p = h->h_addr_list; *p != NULL; ++p)
	++n;
    res->h_addr_list = malloc ((n + 1) * sizeof(*res->h_addr_list));
    if (res->h_addr_list == NULL) {
	freehostent (res);
	return NULL;
    }
    for (i = 0; i < n + 1; ++i) {
	res->h_addr_list[i] = NULL;
    }
    for (i = 0; i < n; ++i) {
	res->h_addr_list[i] = malloc (h->h_length);
	if (res->h_addr_list[i] == NULL) {
	    freehostent (res);
	    return NULL;
	}
	memcpy (res->h_addr_list[i], h->h_addr_list[i], h->h_length);
    }
    return res;
}

