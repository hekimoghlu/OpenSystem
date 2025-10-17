/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <com_right.h>
#include <roken.h>

#ifdef LIBINTL
#include <libintl.h>
#else
#define dgettext(d,s) (s)
#endif

KRB5_LIB_FUNCTION const char * KRB5_LIB_CALL
com_right(struct et_list *list, long code)
{
    struct et_list *p;
    for (p = list; p; p = p->next)
	if (code >= p->table->base && code < p->table->base + p->table->n_msgs)
	    return p->table->msgs[code - p->table->base];
    return NULL;
}

KRB5_LIB_FUNCTION const char * KRB5_LIB_CALL
com_right_r(struct et_list *list, long code, char *str, size_t len)
{
    struct et_list *p;
    for (p = list; p; p = p->next) {
	if (code >= p->table->base && code < p->table->base + p->table->n_msgs) {
	    const char *msg = p->table->msgs[code - p->table->base];
#ifdef LIBINTL
	    char domain[12 + 20];
	    snprintf(domain, sizeof(domain), "heim_com_err%d", p->table->base);
#endif
	    if (msg == NULL)
		snprintf(str, len, "Reserved error code %ld in base %ld",  
			 code - p->table->base, p->table->base);
	    else
		strlcpy(str, dgettext(domain, msg), len);
	    return str;
	}
    }
    return NULL;
}

struct foobar {
    struct et_list etl;
    struct error_table et;
};

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
initialize_error_table_r(struct et_list **list,
			 const char **messages,
			 int num_errors,
			 long base)
{
    struct et_list *et, **end;
    struct foobar *f;
    for (end = list, et = *list; et; end = &et->next, et = et->next)
        if (et->table->msgs == messages)
            return;
    f = malloc(sizeof(*f));
    if (f == NULL)
        return;
    et = &f->etl;
    et->table = &f->et;
    et->table->msgs = messages;
    et->table->n_msgs = num_errors;
    et->table->base = base;
    et->next = NULL;
    *end = et;
}


KRB5_LIB_FUNCTION void KRB5_LIB_CALL
free_error_table(struct et_list *et)
{
    while(et){
	struct et_list *p = et;
	et = et->next;
	free(p);
    }
}
