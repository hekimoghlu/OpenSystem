/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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
#include "kcm_locl.h"

void
kcm_service(void *ctx, const heim_idata *req,
	    const heim_icred cred,
	    heim_ipc_complete complete,
	    heim_sipc_call cctx)
{
    kcm_client peercred;
    krb5_error_code ret;
    krb5_data request, rep;
    unsigned char *buf;
    size_t len;

    krb5_data_zero(&rep);

    peercred.uid = heim_ipc_cred_get_uid(cred);
    peercred.pid = heim_ipc_cred_get_pid(cred);
    peercred.session = heim_ipc_cred_get_session(cred);
    peercred.audit_token = heim_ipc_cred_get_audit_token(cred);
    peercred.iakerb_access = IAKERB_NOT_CHECKED;
    peercred.execpath[0] = '\0';

    if (req->length < 4) {
	kcm_log(1, "malformed request from process %d (too short)",
		peercred.pid);
	(*complete)(cctx, EINVAL, NULL);
	return;
    }

    buf = req->data;
    len = req->length;

    if (buf[0] != KCM_PROTOCOL_VERSION_MAJOR ||
	buf[1] != KCM_PROTOCOL_VERSION_MINOR) {
	kcm_log(1, "incorrect protocol version %d.%d from process %d",
		buf[0], buf[1], peercred.pid);
	(*complete)(cctx, EINVAL, NULL);
	return;
    }

    request.data = buf + 2;
    request.length = len - 2;

    /* buf is now pointing at opcode */

    ret = kcm_dispatch(kcm_context, &peercred, &request, &rep);

    static int num_req = 0;
    if (num_req++ > max_num_requests)
	heim_sipc_timeout(0);

    (*complete)(cctx, ret, &rep);
    krb5_data_free(&rep);
}
