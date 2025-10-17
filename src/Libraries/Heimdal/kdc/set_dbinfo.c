/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#include "kdc_locl.h"

static krb5_error_code
add_db(krb5_context context, struct krb5_kdc_configuration *c,
       const char *conf, const char *master_key)
{
    krb5_error_code ret;
    void *ptr;

    ptr = realloc(c->db, (c->num_db + 1) * sizeof(*c->db));
    if (ptr == NULL) {
	krb5_set_error_message(context, ENOMEM, "malloc: out of memory");
	return ENOMEM;
    }
    c->db = ptr;

    ret = hdb_create(context, &c->db[c->num_db], conf);
    if(ret)
	return ret;

    c->num_db++;

    if (master_key) {
	ret = hdb_set_master_keyfile(context, c->db[c->num_db - 1], master_key);
	if (ret)
	    return ret;
    }

    return 0;
}

krb5_error_code
krb5_kdc_set_dbinfo(krb5_context context, struct krb5_kdc_configuration *c)
{
    struct hdb_dbinfo *info, *d;
    krb5_error_code ret;
    unsigned i;

    /* fetch the databases */
    ret = hdb_get_dbinfo(context, &info);
    if (ret)
	return ret;

    d = NULL;
    while ((d = hdb_dbinfo_get_next(info, d)) != NULL) {

	ret = add_db(context, c,
		     hdb_dbinfo_get_dbname(context, d),
		     hdb_dbinfo_get_mkey_file(context, d));
	if (ret)
	    goto out;

	kdc_log(context, c, 0, "label: %s",
		hdb_dbinfo_get_label(context, d));
	kdc_log(context, c, 0, "\tdbname: %s",
		hdb_dbinfo_get_dbname(context, d));
	kdc_log(context, c, 0, "\tmkey_file: %s",
		hdb_dbinfo_get_mkey_file(context, d));
	kdc_log(context, c, 0, "\tacl_file: %s",
		hdb_dbinfo_get_acl_file(context, d));
    }
    hdb_free_dbinfo(context, &info);

    return 0;
out:
    for (i = 0; i < c->num_db; i++)
	if (c->db[i] && c->db[i]->hdb_destroy)
	    (*c->db[i]->hdb_destroy)(context, c->db[i]);
    c->num_db = 0;
    free(c->db);
    c->db = NULL;

    hdb_free_dbinfo(context, &info);

    return ret;
}


