/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
#include "der_locl.h"
#include <hex.h>

RCSID("$Id$");

int
der_parse_hex_heim_integer (const char *p, heim_integer *data)
{
    ssize_t len;

    data->length = 0;
    data->negative = 0;
    data->data = NULL;

    if (*p == '-') {
	p++;
	data->negative = 1;
    }

    len = strlen(p);
    if (len <= 0) {
	data->data = NULL;
	data->length = 0;
	return EINVAL;
    }

    data->length = (len / 2) + 1;
    data->data = malloc(data->length);
    if (data->data == NULL) {
	data->length = 0;
	return ENOMEM;
    }

    len = hex_decode(p, data->data, data->length);
    if (len < 0) {
	free(data->data);
	data->data = NULL;
	data->length = 0;
	return EINVAL;
    }

    {
	unsigned char *q = data->data;
	while(len > 0 && *q == 0) {
	    q++;
	    len--;
	}
	data->length = len;
	memmove(data->data, q, len);
    }
    return 0;
}

int
der_print_hex_heim_integer (const heim_integer *data, char **p)
{
    ssize_t len;
    char *q;

    len = hex_encode(data->data, data->length, p);
    if (len < 0)
	return ENOMEM;

    if (data->negative) {
	len = asprintf(&q, "-%s", *p);
	free(*p);
	if (len < 0)
	    return ENOMEM;
	*p = q;
    }
    return 0;
}

int
der_print_heim_oid (const heim_oid *oid, char delim, char **str)
{
    struct rk_strpool *p = NULL;
    size_t i;

    if (oid->length == 0)
	return EINVAL;

    for (i = 0; i < oid->length ; i++) {
	p = rk_strpoolprintf(p, "%d", oid->components[i]);
	if (p && i < oid->length - 1)
	    p = rk_strpoolprintf(p, "%c", delim);
	if (p == NULL) {
	    *str = NULL;
	    return ENOMEM;
	}
    }

    *str = rk_strpoolcollect(p);
    if (*str == NULL)
	return ENOMEM;
    return 0;
}

int
der_parse_heim_oid (const char *str, const char *sep, heim_oid *data)
{
    char *s, *w, *brkt, *endptr;
    unsigned int *c;
    long l;

    data->length = 0;
    data->components = NULL;

    if (sep == NULL)
	sep = ".";

    s = strdup(str);

    for (w = strtok_r(s, sep, &brkt);
	 w != NULL;
	 w = strtok_r(NULL, sep, &brkt)) {

	c = realloc(data->components,
		    (data->length + 1) * sizeof(data->components[0]));
	if (c == NULL) {
	    der_free_oid(data);
	    free(s);
	    return ENOMEM;
	}
	data->components = c;

	l = strtol(w, &endptr, 10);
	if (*endptr != '\0' || l < 0 || l > INT_MAX) {
	    der_free_oid(data);
	    free(s);
	    return EINVAL;
	}
	data->components[data->length++] = (unsigned int)l;
    }
    free(s);
    return 0;
}
