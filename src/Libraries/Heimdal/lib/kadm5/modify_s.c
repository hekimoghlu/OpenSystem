/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 25, 2024.
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
modify_principal(void *server_handle,
		 kadm5_principal_ent_t princ,
		 uint32_t mask,
		 uint32_t forbidden_mask)
{
    kadm5_server_context *context = server_handle;
    hdb_entry_ex ent;
    kadm5_ret_t ret;

    memset(&ent, 0, sizeof(ent));

    if((mask & forbidden_mask))
	return KADM5_BAD_MASK;
    if((mask & KADM5_POLICY) && strcmp(princ->policy, "default"))
	return KADM5_UNK_POLICY;

    if (!context->keep_open) {
	ret = context->db->hdb_open(context->context, context->db, O_RDWR, 0);
	if(ret)
	    return ret;
    }
    ret = context->db->hdb_fetch_kvno(context->context, context->db,
				      princ->principal, HDB_F_GET_ANY|HDB_F_ADMIN_DATA, 0, &ent);
    if(ret)
	goto out;
    ret = _kadm5_setup_entry(context, &ent, mask, princ, mask, NULL, 0);
    if(ret)
	goto out2;
    ret = _kadm5_set_modifier(context, &ent.entry);
    if(ret)
	goto out2;

    ret = hdb_seal_keys(context->context, context->db, &ent.entry);
    if (ret)
	goto out2;

    if((mask & KADM5_POLICY)) {
	HDB_extension ext;

	ext.data.element = choice_HDB_extension_data_policy;
	ext.data.u.policy = strdup(princ->policy);
	if (ext.data.u.policy == NULL) {
	    ret = ENOMEM;
	    goto out2;
	}
	/* This calls free_HDB_extension(), freeing ext.data.u.policy */
	ret = hdb_replace_extension(context->context, &ent.entry, &ext);
	if (ret)
	    goto out2;
    }

    ret = context->db->hdb_store(context->context, context->db,
			     HDB_F_REPLACE, &ent);
    if (ret)
	goto out2;

    kadm5_log_modify (context,
		      &ent.entry,
		      mask | KADM5_MOD_NAME | KADM5_MOD_TIME);

out2:
    hdb_free_entry(context->context, &ent);
out:
    if (!context->keep_open)
	context->db->hdb_close(context->context, context->db);
    return _kadm5_error_code(ret);
}


kadm5_ret_t
kadm5_s_modify_principal(void *server_handle,
			 kadm5_principal_ent_t princ,
			 uint32_t mask)
{
    return modify_principal(server_handle, princ, mask,
			    KADM5_LAST_PWD_CHANGE | KADM5_MOD_TIME
			    | KADM5_MOD_NAME | KADM5_MKVNO
			    | KADM5_AUX_ATTRIBUTES | KADM5_LAST_SUCCESS
			    | KADM5_LAST_FAILED);
}
