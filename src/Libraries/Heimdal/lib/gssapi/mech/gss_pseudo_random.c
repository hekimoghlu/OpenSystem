/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
/* $Id$ */

#include "mech_locl.h"

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_pseudo_random(OM_uint32 *__nonnull minor_status,
		  __nonnull gss_ctx_id_t context,
		  int prf_key,
		  __nonnull const gss_buffer_t prf_in,
		  ssize_t desired_output_len,
		  __nonnull gss_buffer_t prf_out)
{
    struct _gss_context *ctx = (struct _gss_context *) context;
    gssapi_mech_interface m;
    OM_uint32 major_status;

    _mg_buffer_zero(prf_out);
    *minor_status = 0;

    if (ctx == NULL) {
	*minor_status = 0;
	return GSS_S_NO_CONTEXT;
    }

    m = ctx->gc_mech;

    if (m->gm_pseudo_random == NULL)
	return GSS_S_UNAVAILABLE;

    major_status = (*m->gm_pseudo_random)(minor_status, ctx->gc_ctx,
					  prf_key, prf_in, desired_output_len,
					  prf_out);
    if (major_status != GSS_S_COMPLETE)
	_gss_mg_error(m, *minor_status);

    return major_status;
}
