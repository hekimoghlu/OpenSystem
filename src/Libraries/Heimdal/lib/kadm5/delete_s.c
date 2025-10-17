/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
kadm5_s_delete_principal(void *server_handle, krb5_principal princ)
{
    kadm5_server_context *context = server_handle;
    kadm5_ret_t ret;
    hdb_entry_ex ent;

    memset(&ent, 0, sizeof(ent));
    if (!context->keep_open) {
	ret = context->db->hdb_open(context->context, context->db, O_RDWR, 0);
	if(ret) {
	    krb5_warn(context->context, ret, "opening database");
	    return ret;
	}
    }
    ret = context->db->hdb_fetch_kvno(context->context, context->db, princ,
				      HDB_F_DECRYPT|HDB_F_GET_ANY|HDB_F_ADMIN_DATA, 0, &ent);
    if(ret == HDB_ERR_NOENTRY)
	goto out;
    if(ent.entry.flags.immutable) {
	ret = KADM5_PROTECT_PRINCIPAL;
	goto out2;
    }

    ret = hdb_seal_keys(context->context, context->db, &ent.entry);
    if (ret)
	goto out2;

    ret = context->db->hdb_remove(context->context, context->db, princ);
    if (ret)
	goto out2;

    kadm5_log_delete (context, princ);

out2:
    hdb_free_entry(context->context, &ent);
out:
    if (!context->keep_open)
	context->db->hdb_close(context->context, context->db);
    return _kadm5_error_code(ret);
}
