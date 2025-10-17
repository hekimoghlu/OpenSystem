/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#include "gsskrb5_locl.h"

#include <roken.h>

OM_uint32 GSSAPI_CALLCONV
_gk_wrap_iov(OM_uint32 * minor_status,
	     gss_ctx_id_t  context_handle,
	     int conf_req_flag,
	     gss_qop_t qop_req,
	     int * conf_state,
	     gss_iov_buffer_desc *iov,
	     int iov_count)
{
  const gsskrb5_ctx ctx = (const gsskrb5_ctx) context_handle;
  krb5_context context;

  GSSAPI_KRB5_INIT (&context);

  if (ctx->more_flags & IS_CFX)
      return _gssapi_wrap_cfx_iov(minor_status, &ctx->gk5c, context,
				  conf_req_flag, conf_state,
				  iov, iov_count);

    return GSS_S_FAILURE;
}

OM_uint32 GSSAPI_CALLCONV
_gk_unwrap_iov(OM_uint32 *minor_status,
	       gss_ctx_id_t context_handle,
	       int *conf_state,
	       gss_qop_t *qop_state,
	       gss_iov_buffer_desc *iov,
	       int iov_count)
{
    const gsskrb5_ctx ctx = (const gsskrb5_ctx) context_handle;
    krb5_context context;

    GSSAPI_KRB5_INIT (&context);

    if (ctx->more_flags & IS_CFX)
	return _gssapi_unwrap_cfx_iov(minor_status, &ctx->gk5c, context,
				      conf_state, qop_state, iov, iov_count);

    return GSS_S_FAILURE;
}

OM_uint32 GSSAPI_CALLCONV
_gk_wrap_iov_length(OM_uint32 * minor_status,
		    gss_ctx_id_t context_handle,
		    int conf_req_flag,
		    gss_qop_t qop_req,
		    int *conf_state,
		    gss_iov_buffer_desc *iov,
		    int iov_count)
{
    const gsskrb5_ctx ctx = (const gsskrb5_ctx) context_handle;
    krb5_context context;

    GSSAPI_KRB5_INIT (&context);

    if (ctx->more_flags & IS_CFX)
	return _gssapi_wrap_iov_length_cfx(minor_status, &ctx->gk5c, context,
					   conf_req_flag, qop_req, conf_state,
					   iov, iov_count);

    return GSS_S_FAILURE;
}
