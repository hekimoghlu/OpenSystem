/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_inquire_sec_context_by_oid (OM_uint32 *__nonnull minor_status,
	                        __nonnull const gss_ctx_id_t context_handle,
	                        __nonnull const gss_OID desired_object,
	                        __nonnull gss_buffer_set_t * __nullable data_set)
{
	struct _gss_context	*ctx = (struct _gss_context *) context_handle;
	OM_uint32		major_status;
	gssapi_mech_interface	m;

	*minor_status = 0;
	*data_set = GSS_C_NO_BUFFER_SET;
	if (ctx == NULL)
		return GSS_S_NO_CONTEXT;

	/*
	 * select the approprate underlying mechanism routine and
	 * call it.
	 */

	m = ctx->gc_mech;

	if (m == NULL)
		return GSS_S_BAD_MECH;

	if (m->gm_inquire_sec_context_by_oid != NULL) {
		major_status = m->gm_inquire_sec_context_by_oid(minor_status,
		    ctx->gc_ctx, desired_object, data_set);
		if (major_status != GSS_S_COMPLETE)
			_gss_mg_error(m, *minor_status);
	} else
		major_status = GSS_S_BAD_MECH;

	return major_status;
}

