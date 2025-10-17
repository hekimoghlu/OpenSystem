/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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

kadm5_ret_t
kadm5_s_rename_principal(void *server_handle,
			 krb5_principal source,
			 krb5_principal target)
{
    kadm5_server_context *context = server_handle;
    kadm5_ret_t ret;
    hdb_entry_ex ent;
    krb5_principal oldname;

    memset(&ent, 0, sizeof(ent));
    if(krb5_principal_compare(context->context, source, target))
	return KADM5_DUP; /* XXX is this right? */
    if (!context->keep_open) {
	ret = context->db->hdb_open(context->context, context->db, O_RDWR, 0);
	if(ret)
	    return ret;
    }
    ret = context->db->hdb_fetch_kvno(context->context, context->db,
				      source, HDB_F_GET_ANY|HDB_F_ADMIN_DATA, 0, &ent);
    if(ret){
	if (!context->keep_open)
	    context->db->hdb_close(context->context, context->db);
	goto out;
    }
    ret = _kadm5_set_modifier(context, &ent.entry);
    if(ret)
	goto out2;
    {
	/* fix salt */
	size_t i;
	Salt salt;
	krb5_salt salt2;
	memset(&salt, 0, sizeof(salt));
	krb5_get_pw_salt(context->context, source, &salt2);
	salt.type = hdb_pw_salt;
	salt.salt = salt2.saltvalue;
	for(i = 0; i < ent.entry.keys.len; i++){
	    if(ent.entry.keys.val[i].salt == NULL){
		ent.entry.keys.val[i].salt =
		    malloc(sizeof(*ent.entry.keys.val[i].salt));
		if(ent.entry.keys.val[i].salt == NULL)
		    return ENOMEM;
		ret = copy_Salt(&salt, ent.entry.keys.val[i].salt);
		if(ret)
		    break;
	    }
	}
	krb5_free_salt(context->context, salt2);
    }
    if(ret)
	goto out2;
    oldname = ent.entry.principal;
    ent.entry.principal = target;

    ret = hdb_seal_keys(context->context, context->db, &ent.entry);
    if (ret) {
	ent.entry.principal = oldname;
	goto out2;
    }

    kadm5_log_rename (context, source, &ent.entry);

    ret = context->db->hdb_store(context->context, context->db, 0, &ent);
    if(ret){
	ent.entry.principal = oldname;
	goto out2;
    }
    ret = context->db->hdb_remove(context->context, context->db, oldname);
    ent.entry.principal = oldname;
out2:
    if (!context->keep_open)
	context->db->hdb_close(context->context, context->db);
    hdb_free_entry(context->context, &ent);
out:
    return _kadm5_error_code(ret);
}

