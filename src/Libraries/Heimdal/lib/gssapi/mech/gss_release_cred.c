/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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

/**
 * Release a credentials
 *
 * Its ok to release the GSS_C_NO_CREDENTIAL/NULL credential, it will
 * return a GSS_S_COMPLETE error code. On return cred_handle is set ot
 * GSS_C_NO_CREDENTIAL.
 *
 * Example:
 *
 * @code
 * gss_cred_id_t cred = GSS_C_NO_CREDENTIAL;
 * major = gss_release_cred(&minor, &cred);
 * @endcode
 *
 * @param minor_status minor status return code, mech specific
 * @param cred_handle a pointer to the credential too release
 *
 * @return an gssapi error code
 *
 * @ingroup gssapi
 */

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_release_cred(OM_uint32 *__nonnull minor_status,
		 __nullable gss_cred_id_t * __nonnull cred_handle)
{
	struct _gss_cred *cred = (struct _gss_cred *) *cred_handle;

	if (cred == NULL)
	    return (GSS_S_COMPLETE);

	_gss_mg_release_cred(cred);

	*minor_status = 0;
	*cred_handle = GSS_C_NO_CREDENTIAL;
	return (GSS_S_COMPLETE);
}
