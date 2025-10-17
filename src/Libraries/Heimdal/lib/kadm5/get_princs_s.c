/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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
#include "kadm5_locl.h"

RCSID("$Id$");

struct foreach_data {
    const char *exp;
    char *exp2;
    char **princs;
    int count;
};

static krb5_error_code
add_princ(struct foreach_data *d, char *princ)
{
    char **tmp;
    tmp = realloc(d->princs, (d->count + 1) * sizeof(*tmp));
    if(tmp == NULL)
	return ENOMEM;
    d->princs = tmp;
    d->princs[d->count++] = princ;
    return 0;
}

static krb5_error_code
foreach(krb5_context context, HDB *db, hdb_entry_ex *ent, void *data)
{
    struct foreach_data *d = data;
    char *princ;
    krb5_error_code ret;
    ret = krb5_unparse_name(context, ent->entry.principal, &princ);
    if(ret)
	return ret;
    if(d->exp){
	if(fnmatch(d->exp, princ, 0) == 0 || fnmatch(d->exp2, princ, 0) == 0)
	    ret = add_princ(d, princ);
	else
	    free(princ);
    }else{
	ret = add_princ(d, princ);
    }
    if(ret)
	free(princ);
    return ret;
}

kadm5_ret_t
kadm5_s_get_principals(void *server_handle,
		       const char *expression,
		       char ***princs,
		       int *count)
{
    struct foreach_data d;
    kadm5_server_context *context = server_handle;
    kadm5_ret_t ret;

    if (!context->keep_open) {
	ret = context->db->hdb_open(context->context, context->db, O_RDWR, 0);
	if(ret) {
	    krb5_warn(context->context, ret, "opening database");
	    return ret;
	}
    }
    d.exp = expression;
    {
	krb5_realm r;
	krb5_get_default_realm(context->context, &r);
	asprintf(&d.exp2, "%s@%s", expression, r);
	free(r);
    }
    d.princs = NULL;
    d.count = 0;
    ret = hdb_foreach(context->context, context->db, HDB_F_ADMIN_DATA, foreach, &d);
    if (!context->keep_open)
	context->db->hdb_close(context->context, context->db);
    if(ret == 0)
	ret = add_princ(&d, NULL);
    if(ret == 0){
	*princs = d.princs;
	*count = d.count - 1;
    }else
	kadm5_free_name_list(context, d.princs, &d.count);
    free(d.exp2);
    return _kadm5_error_code(ret);
}
