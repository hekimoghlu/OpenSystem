/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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
#include "mech_locl.h"
#include <heim_threads.h>

#ifdef __BLOCKS__
#include <Block.h>
#endif

#ifdef __BLOCKS__

static void 
complete_block(void *ctx, OM_uint32 maj_stat,
	       gss_status_id_t status, gss_cred_id_t cred,
	       gss_OID_set set, OM_uint32 min_time)
{
    gss_acquire_cred_complete complete = ctx;

    complete(status, cred, set, min_time);
    Block_release(complete);
}

OM_uint32 GSSAPI_LIB_FUNCTION
gss_acquire_cred_ex(__nonnull const gss_name_t desired_name,
		    OM_uint32 flags,
		    OM_uint32 time_req,
		    __nonnull gss_const_OID desired_mech,
		    gss_cred_usage_t cred_usage,
		    __nonnull gss_auth_identity_t identity,
		    __nonnull gss_acquire_cred_complete complete)
{
    OM_uint32 ret;

    complete = (gss_acquire_cred_complete)Block_copy(complete);

    ret = gss_acquire_cred_ex_f(NULL,
				desired_name,
				flags,
				time_req,
				desired_mech,
				cred_usage,
				identity,
				complete,
				complete_block);
    if (ret != GSS_S_COMPLETE)
	Block_release(complete);
    return ret;
}
#endif


OM_uint32 GSSAPI_LIB_FUNCTION
gss_acquire_cred_ex_f(__nullable gss_status_id_t status,
		      __nullable gss_name_t desired_name,
		      OM_uint32 flags,
		      OM_uint32 time_req,
		      __nonnull gss_const_OID desired_mech,
		      gss_cred_usage_t cred_usage,
		      __nonnull gss_auth_identity_t identity,
		       void * __nullable userctx,
		      void (*__nonnull usercomplete)(void *__nullable, OM_uint32, gss_status_id_t __nullable, gss_cred_id_t __nullable, gss_OID_set __nullable, OM_uint32))
{
        OM_uint32 major_status, minor_status;
	gss_name_t name = GSS_C_NO_NAME;
	gss_cred_id_t cred;
	OM_uint32 junk;
	gss_buffer_desc buffer;

	if (usercomplete == NULL)
	    return GSS_S_CALL_INACCESSIBLE_READ;

	/*
	 * If no desired_name, make one up from the identity
	 */
	if (desired_name == GSS_C_NO_NAME) {
	    char *str;
	    if (identity->username == NULL)
		return GSS_S_FAILURE;
	    if (identity->realm)
		asprintf(&str, "%s@%s", identity->username, identity->realm);
	    else
		str = strdup(identity->username);
	    buffer.value = str;
	    buffer.length = strlen(str);
	    
	    major_status = gss_import_name(&minor_status, &buffer, GSS_C_NT_USER_NAME, &name);
	    free(str);
	    if (major_status)
		return major_status;

	    desired_name = name;
	}

	/*
	 * First make sure that at least one of the requested
	 * mechanisms is one that we support.
	 */
	if (desired_mech) {
		int t;
		gss_test_oid_set_member(&junk, desired_mech, _gss_mech_oids, &t);
		if (!t) {
			if (name)
				gss_release_name(&junk, &name);
			return (GSS_S_BAD_MECH);
		}
	}

	buffer.value = identity->password;
	buffer.length = strlen(identity->password);

	cred = GSS_C_NO_CREDENTIAL;

	major_status = gss_acquire_cred_ext(&minor_status,
					    desired_name,
					    GSS_C_CRED_PASSWORD,
					    &buffer,
					    time_req,
					    desired_mech,
					    cred_usage,
					    &cred);
	if (name)
		gss_release_name(&junk, &name);
	if (major_status)
		return major_status;

	usercomplete(userctx, major_status, status, 
		     cred, GSS_C_NO_OID_SET, GSS_C_INDEFINITE);

	return GSS_S_COMPLETE;
}
