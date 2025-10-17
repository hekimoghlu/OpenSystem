/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include <gssapi_spi.h>
#include <heim_threads.h>

/**
 * Destroy a credential 
 *
 * gss_release_cred() frees the memory, gss_destroy_cred() removes the credentials from memory/disk and then call gss_release_cred() on the credential.
 *
 * @param min_stat minor status code
 * @param cred_handle credentail to destory
 *
 * @returns a gss_error code, see gss_display_status() about printing
 *          the error code.
 * 
 * @ingroup gssapi
 */

OM_uint32 GSSAPI_LIB_FUNCTION
gss_destroy_cred(OM_uint32 *__nonnull min_stat,
		 __nullable gss_cred_id_t * __nonnull cred_handle)
{
    struct _gss_cred *cred;
    struct _gss_mechanism_cred *mc;
    OM_uint32 junk;

    if (cred_handle == NULL)
	return GSS_S_CALL_INACCESSIBLE_READ;
    if (*cred_handle == GSS_C_NO_CREDENTIAL)
	return GSS_S_COMPLETE;

    cred = (struct _gss_cred *)*cred_handle;
    *cred_handle = GSS_C_NO_CREDENTIAL;

    while (HEIM_SLIST_FIRST(&cred->gc_mc)) {
	mc = HEIM_SLIST_FIRST(&cred->gc_mc);
	HEIM_SLIST_REMOVE_HEAD(&cred->gc_mc, gmc_link);

	if (mc->gmc_mech->gm_destroy_cred)
	    mc->gmc_mech->gm_destroy_cred(&junk, &mc->gmc_cred);
	else
	    mc->gmc_mech->gm_release_cred(&junk, &mc->gmc_cred);
	free(mc);
    }
    free(cred);

    return GSS_S_COMPLETE;
}
