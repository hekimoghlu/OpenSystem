/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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

static kadm5_ret_t
change(void *server_handle,
       krb5_principal princ,
       int keepold,
       const char *password,
       int cond,
       int n_ks_tuple,
       krb5_key_salt_tuple *ks_tuple)
{
    kadm5_server_context *context = server_handle;
    hdb_entry_ex ent;
    kadm5_ret_t ret;
    Key *keys;
    size_t num_keys;
    int existsp = 0;
    int flags;
    unsigned int modify_flags;

    memset(&ent, 0, sizeof(ent));
    if (!context->keep_open) {
	ret = context->db->hdb_open(context->context, context->db, O_RDWR, 0);
	if(ret)
	    return ret;
    }

    flags = HDB_F_GET_ANY|HDB_F_ADMIN_DATA;
    if (cond)
	flags |= HDB_F_DECRYPT;

    ret = context->db->hdb_fetch_kvno(context->context, context->db, princ, flags, 0, &ent);
    if(ret)
	goto out;
    
    modify_flags = KADM5_PRINCIPAL | KADM5_MOD_NAME | KADM5_MOD_TIME |
	KADM5_KVNO | KADM5_PW_EXPIRATION | KADM5_TL_DATA;

    if (keepold || cond) {
	/*
	 * We save these for now so we can handle password history checking;
	 * we handle keepold further below.
	 */
	ret = hdb_add_current_keys_to_history(context->context, &ent.entry);
	if (ret)
	    goto out;
    }

    if (context->db->hdb_capability_flags & HDB_CAP_F_HANDLE_PASSWORDS) {
	ret = context->db->hdb_password(context->context, context->db,
					&ent, password,
					cond ? HDB_PWD_CONDITIONAL : 0);
	if (ret)
	    goto out2;
    } else {
	
	num_keys = ent.entry.keys.len;
	keys     = ent.entry.keys.val;

	ent.entry.keys.len = 0;
	ent.entry.keys.val = NULL;
	
	ret = _kadm5_set_keys(context, &ent.entry, password,
			      n_ks_tuple, ks_tuple);
	if(ret) {
	    _kadm5_free_keys(context->context, (int)num_keys, keys);
	    goto out2;
	}
	_kadm5_free_keys(context->context, (int)num_keys, keys);

	if (cond) {
	    HDB_extension *ext;

	    ext = hdb_find_extension(&ent.entry, choice_HDB_extension_data_hist_keys);
	    if (ext != NULL)
		existsp = _kadm5_exists_keys_hist(ent.entry.keys.val,
						  ent.entry.keys.len,
						  &ext->data.u.hist_keys);
	}

	if (existsp) {
	    ret = KADM5_PASS_REUSE;
	    krb5_set_error_message(context->context, ret,
				   "Password reuse forbidden");
	    goto out2;
	}
    }
    ent.entry.kvno++;

    modify_flags |= KADM5_KEY_DATA;
	
    if (keepold) {
	ret = hdb_seal_keys(context->context, context->db, &ent.entry);
	if (ret)
	    goto out2;
    } else {
	HDB_extension ext;

	ext.data.element = choice_HDB_extension_data_hist_keys;
	ext.data.u.hist_keys.len = 0;
	ext.data.u.hist_keys.val = NULL;
	ret = hdb_replace_extension(context->context, &ent.entry, &ext);
	if (ret)
	    goto out2;
    }

    ret = _kadm5_set_modifier(context, &ent.entry);
    if(ret)
	goto out2;

    ret = _kadm5_bump_pw_expire(context, &ent.entry);
    if (ret)
	goto out2;

    ret = context->db->hdb_store(context->context, context->db,
				 HDB_F_REPLACE|HDB_F_CHANGE_PASSWORD, &ent);
    if (ret)
	goto out2;

    kadm5_log_modify(context, &ent.entry, modify_flags);

out2:
    hdb_free_entry(context->context, &ent);
out:
    if (!context->keep_open)
	context->db->hdb_close(context->context, context->db);
    return _kadm5_error_code(ret);
}



/*
 * change the password of `princ' to `password' if it's not already that.
 */

kadm5_ret_t
kadm5_s_chpass_principal_cond(void *server_handle,
			      krb5_principal princ,
			      int keepold,
			      const char *password,
			      int n_ks_tuple,
			      krb5_key_salt_tuple *ks_tuple)
{
    return change (server_handle, princ, keepold, password, 1, n_ks_tuple, ks_tuple);
}

/*
 * change the password of `princ' to `password'
 */

kadm5_ret_t
kadm5_s_chpass_principal(void *server_handle,
			 krb5_principal princ,
			 int keepold,
			 const char *password,
			 int n_ks_tuple,
			 krb5_key_salt_tuple *ks_tuple)
{
    return change (server_handle, princ, keepold, password, 0, n_ks_tuple, ks_tuple);
}

/*
 * change keys for `princ' to `keys'
 */

kadm5_ret_t
kadm5_s_chpass_principal_with_key(void *server_handle,
				  krb5_principal princ,
				  int keepold,
				  int n_key_data,
				  krb5_key_data *key_data)
{
    kadm5_server_context *context = server_handle;
    hdb_entry_ex ent;
    kadm5_ret_t ret;

    memset(&ent, 0, sizeof(ent));
    if (!context->keep_open) {
	ret = context->db->hdb_open(context->context, context->db, O_RDWR, 0);
	if(ret)
	    return ret;
    }
    ret = context->db->hdb_fetch_kvno(context->context, context->db, princ, 0,
				      HDB_F_GET_ANY|HDB_F_ADMIN_DATA, &ent);
    if(ret == HDB_ERR_NOENTRY)
	goto out;
    if (keepold) {
	ret = hdb_add_current_keys_to_history(context->context, &ent.entry);
	if (ret)
	    goto out2;
    }
    ret = _kadm5_set_keys2(context, &ent.entry, n_key_data, key_data);
    if(ret)
	goto out2;
    ent.entry.kvno++;
    ret = _kadm5_set_modifier(context, &ent.entry);
    if(ret)
	goto out2;
    ret = _kadm5_bump_pw_expire(context, &ent.entry);
    if (ret)
	goto out2;

    if (keepold) {
	ret = hdb_seal_keys(context->context, context->db, &ent.entry);
	if (ret)
	    goto out2;
    } else {
	HDB_extension ext;

	ext.data.element = choice_HDB_extension_data_hist_keys;
	ext.data.u.hist_keys.len = 0;
	ext.data.u.hist_keys.val = NULL;
	hdb_replace_extension(context->context, &ent.entry, &ext);
    }


    ret = context->db->hdb_store(context->context, context->db,
				 HDB_F_REPLACE, &ent);
    if (ret)
	goto out2;

    kadm5_log_modify (context,
		      &ent.entry,
		      KADM5_PRINCIPAL | KADM5_MOD_NAME | KADM5_MOD_TIME |
		      KADM5_KEY_DATA | KADM5_KVNO | KADM5_PW_EXPIRATION |
		      KADM5_TL_DATA);

out2:
    hdb_free_entry(context->context, &ent);
out:
    if (!context->keep_open)
	context->db->hdb_close(context->context, context->db);
    return _kadm5_error_code(ret);
}
