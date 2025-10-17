/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#include "hdb_locl.h"
#include <assert.h>

typedef struct {
    char *path;
    krb5_keytab keytab;
} *hdb_keytab;

/*
 *
 */

static krb5_error_code
hkt_close(krb5_context context, HDB *db)
{
    hdb_keytab k = (hdb_keytab)db->hdb_db;
    krb5_error_code ret;

    assert(k->keytab);

    ret = krb5_kt_close(context, k->keytab);
    k->keytab = NULL;

    return ret;
}

static krb5_error_code
hkt_destroy(krb5_context context, HDB *db)
{
    hdb_keytab k = (hdb_keytab)db->hdb_db;
    krb5_error_code ret;

    ret = hdb_clear_master_key (context, db);

    free(k->path);
    free(k);

    free(db->hdb_name);
    free(db);
    return ret;
}

static krb5_error_code
hkt_lock(krb5_context context, HDB *db, int operation)
{
    return 0;
}

static krb5_error_code
hkt_unlock(krb5_context context, HDB *db)
{
    return 0;
}

static krb5_error_code
hkt_firstkey(krb5_context context, HDB *db,
	     unsigned flags, hdb_entry_ex *entry)
{
    return HDB_ERR_DB_INUSE;
}

static krb5_error_code
hkt_nextkey(krb5_context context, HDB * db, unsigned flags,
	     hdb_entry_ex * entry)
{
    return HDB_ERR_DB_INUSE;
}

static krb5_error_code
hkt_open(krb5_context context, HDB * db, int flags, mode_t mode)
{
    hdb_keytab k = (hdb_keytab)db->hdb_db;
    krb5_error_code ret;

    assert(k->keytab == NULL);

    ret = krb5_kt_resolve(context, k->path, &k->keytab);
    if (ret)
	return ret;

    return 0;
}

static krb5_error_code
hkt_fetch_kvno(krb5_context context, HDB * db, krb5_const_principal principal,
	       unsigned flags, krb5_kvno kvno, hdb_entry_ex * entry)
{
    hdb_keytab k = (hdb_keytab)db->hdb_db;
    krb5_error_code ret;
    krb5_keytab_entry ktentry;

    if (!(flags & HDB_F_KVNO_SPECIFIED)) {
	    /* Preserve previous behaviour if no kvno specified */
	    kvno = 0;
    }

    memset(&ktentry, 0, sizeof(ktentry));

    entry->entry.flags.server = 1;
    entry->entry.flags.forwardable = 1;
    entry->entry.flags.renewable = 1;

    /* Not recorded in the OD backend, make something up */
    ret = krb5_parse_name(context, "hdb/keytab@WELL-KNOWN:KEYTAB-BACKEND",
			  &entry->entry.created_by.principal);
    if (ret)
	goto out;

    /*
     * XXX really needs to try all enctypes and just not pick the
     * first one, even if that happens to be des3-cbc-sha1 (ie best
     * enctype) in the Apple case. A while loop over all known
     * enctypes should work.
     */

    ret = krb5_kt_get_entry(context, k->keytab, principal, kvno, 0, &ktentry);
    if (ret) {
	ret = HDB_ERR_NOENTRY;
	goto out;
    }

    ret = krb5_copy_principal(context, principal, &entry->entry.principal);
    if (ret)
	goto out;

    ret = _hdb_keytab2hdb_entry(context, &ktentry, entry);

 out:
    if (ret) {
	free_hdb_entry(&entry->entry);
	memset(&entry->entry, 0, sizeof(entry->entry));
    }
    krb5_kt_free_entry(context, &ktentry);

    return ret;
}

static krb5_error_code
hkt_store(krb5_context context, HDB * db, unsigned flags,
	  hdb_entry_ex * entry)
{
    return HDB_ERR_DB_INUSE;
}


krb5_error_code
hdb_keytab_create(krb5_context context, HDB ** db, const char *arg)
{
    hdb_keytab k;

    *db = calloc(1, sizeof(**db));
    if (*db == NULL) {
	krb5_set_error_message(context, ENOMEM, "malloc: out of memory");
	return ENOMEM;
    }
    memset(*db, 0, sizeof(**db));

    k = calloc(1, sizeof(*k));
    if (k == NULL) {
	free(*db);
	*db = NULL;
	krb5_set_error_message(context, ENOMEM, "malloc: out of memory");
	return ENOMEM;
    }

    k->path = strdup(arg);
    if (k->path == NULL) {
	free(k);
	free(*db);
	*db = NULL;
	krb5_set_error_message(context, ENOMEM, "malloc: out of memory");
	return ENOMEM;
    }


    (*db)->hdb_db = k;

    (*db)->hdb_master_key_set = 0;
    (*db)->hdb_openp = 0;
    (*db)->hdb_open = hkt_open;
    (*db)->hdb_close = hkt_close;
    (*db)->hdb_fetch_kvno = hkt_fetch_kvno;
    (*db)->hdb_store = hkt_store;
    (*db)->hdb_remove = NULL;
    (*db)->hdb_firstkey = hkt_firstkey;
    (*db)->hdb_nextkey = hkt_nextkey;
    (*db)->hdb_lock = hkt_lock;
    (*db)->hdb_unlock = hkt_unlock;
    (*db)->hdb_rename = NULL;
    (*db)->hdb__get = NULL;
    (*db)->hdb__put = NULL;
    (*db)->hdb__del = NULL;
    (*db)->hdb_destroy = hkt_destroy;

    return 0;
}
