/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
#include "krb5_locl.h"

#ifdef HEIM_KT_MEMORY

/* memory operations -------------------------------------------- */

struct mkt_data {
    krb5_keytab_entry *entries;
    int num_entries;
    char *name;
    int refcount;
    struct mkt_data *next;
};

/* this mutex protects mkt_head, ->refcount, and ->next
 * content is not protected (name is static and need no protection)
 */
static HEIMDAL_MUTEX mkt_mutex = HEIMDAL_MUTEX_INITIALIZER;
static struct mkt_data *mkt_head;


static krb5_error_code KRB5_CALLCONV
mkt_resolve(krb5_context context, const char *name, krb5_keytab id)
{
    struct mkt_data *d;

    HEIMDAL_MUTEX_lock(&mkt_mutex);

    for (d = mkt_head; d != NULL; d = d->next)
	if (strcmp(d->name, name) == 0)
	    break;
    if (d) {
	if (d->refcount < 1)
	    krb5_abortx(context, "Double close on memory keytab, "
			"refcount < 1 %d", d->refcount);
	d->refcount++;
	id->data = d;
	HEIMDAL_MUTEX_unlock(&mkt_mutex);
	return 0;
    }

    d = calloc(1, sizeof(*d));
    if(d == NULL) {
	HEIMDAL_MUTEX_unlock(&mkt_mutex);
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    d->name = strdup(name);
    if (d->name == NULL) {
	HEIMDAL_MUTEX_unlock(&mkt_mutex);
	free(d);
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    d->entries = NULL;
    d->num_entries = 0;
    d->refcount = 1;
    d->next = mkt_head;
    mkt_head = d;
    HEIMDAL_MUTEX_unlock(&mkt_mutex);
    id->data = d;
    return 0;
}

static krb5_error_code KRB5_CALLCONV
mkt_close(krb5_context context, krb5_keytab id)
{
    struct mkt_data *d = id->data, **dp;
    int i;

    HEIMDAL_MUTEX_lock(&mkt_mutex);
    if (d->refcount < 1)
	krb5_abortx(context,
		    "krb5 internal error, memory keytab refcount < 1 on close");

    if (--d->refcount > 0) {
	HEIMDAL_MUTEX_unlock(&mkt_mutex);
	return 0;
    }
    for (dp = &mkt_head; *dp != NULL; dp = &(*dp)->next) {
	if (*dp == d) {
	    *dp = d->next;
	    break;
	}
    }
    HEIMDAL_MUTEX_unlock(&mkt_mutex);

    free(d->name);
    for(i = 0; i < d->num_entries; i++)
	krb5_kt_free_entry(context, &d->entries[i]);
    free(d->entries);
    free(d);
    return 0;
}

static krb5_error_code KRB5_CALLCONV
mkt_get_name(krb5_context context,
	     krb5_keytab id,
	     char *name,
	     size_t namesize)
{
    struct mkt_data *d = id->data;
    strlcpy(name, d->name, namesize);
    return 0;
}

static krb5_error_code KRB5_CALLCONV
mkt_start_seq_get(krb5_context context,
		  krb5_keytab id,
		  krb5_kt_cursor *c)
{
    /* XXX */
    c->fd = 0;
    return 0;
}

static krb5_error_code KRB5_CALLCONV
mkt_next_entry(krb5_context context,
	       krb5_keytab id,
	       krb5_keytab_entry *entry,
	       krb5_kt_cursor *c)
{
    struct mkt_data *d = id->data;
    if(c->fd >= d->num_entries)
	return KRB5_KT_END;
    return krb5_kt_copy_entry_contents(context, &d->entries[c->fd++], entry);
}

static krb5_error_code KRB5_CALLCONV
mkt_end_seq_get(krb5_context context,
		krb5_keytab id,
		krb5_kt_cursor *cursor)
{
    return 0;
}

static krb5_error_code KRB5_CALLCONV
mkt_add_entry(krb5_context context,
	      krb5_keytab id,
	      krb5_keytab_entry *entry)
{
    struct mkt_data *d = id->data;
    krb5_keytab_entry *tmp;
    tmp = realloc(d->entries, (d->num_entries + 1) * sizeof(*d->entries));
    if(tmp == NULL) {
	krb5_set_error_message(context, ENOMEM,
			       N_("malloc: out of memory", ""));
	return ENOMEM;
    }
    d->entries = tmp;
    return krb5_kt_copy_entry_contents(context, entry,
				       &d->entries[d->num_entries++]);
}

static krb5_error_code KRB5_CALLCONV
mkt_remove_entry(krb5_context context,
		 krb5_keytab id,
		 krb5_keytab_entry *entry)
{
    struct mkt_data *d = id->data;
    krb5_keytab_entry *e, *end;
    int found = 0;

    if (d->num_entries == 0) {
	krb5_clear_error_message(context);
        return KRB5_KT_NOTFOUND;
    }

    /* do this backwards to minimize copying */
    for(end = d->entries + d->num_entries, e = end - 1; e >= d->entries; e--) {
	if(krb5_kt_compare(context, e, entry->principal,
			   entry->vno, entry->keyblock.keytype)) {
	    krb5_kt_free_entry(context, e);
	    memmove(e, e + 1, (end - e - 1) * sizeof(*e));
	    memset(end - 1, 0, sizeof(*end));
	    d->num_entries--;
	    end--;
	    found = 1;
	}
    }
    if (!found) {
	krb5_clear_error_message (context);
	return KRB5_KT_NOTFOUND;
    }
    e = realloc(d->entries, d->num_entries * sizeof(*d->entries));
    if(e != NULL || d->num_entries == 0)
	d->entries = e;
    return 0;
}

const krb5_kt_ops krb5_mkt_ops = {
    "MEMORY",
    mkt_resolve,
    mkt_get_name,
    mkt_close,
    NULL, /* destroy */
    NULL, /* get */
    mkt_start_seq_get,
    mkt_next_entry,
    mkt_end_seq_get,
    mkt_add_entry,
    mkt_remove_entry
};

#endif /* HEIM_KT_MEMORY */

