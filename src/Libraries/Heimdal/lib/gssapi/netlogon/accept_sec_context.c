/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#include "netlogon.h"

/*
 * Not implemented: this is needed only by domain controllers.
 */

OM_uint32
_netlogon_accept_sec_context
(OM_uint32 * minor_status,
 gss_ctx_id_t * context_handle,
 const gss_cred_id_t acceptor_cred_handle,
 const gss_buffer_t input_token_buffer,
 const gss_channel_bindings_t input_chan_bindings,
 gss_name_t * src_name,
 gss_OID * mech_type,
 gss_buffer_t output_token,
 OM_uint32 * ret_flags,
 OM_uint32 * time_rec,
 gss_cred_id_t * delegated_cred_handle
    )
{

    output_token->value = NULL;
    output_token->length = 0;

    *minor_status = 0;

    if (context_handle == NULL)
        return GSS_S_FAILURE;

    if (input_token_buffer == GSS_C_NO_BUFFER)
        return GSS_S_FAILURE;

    if (src_name)
        *src_name = GSS_C_NO_NAME;
    if (mech_type)
        *mech_type = GSS_C_NO_OID;
    if (ret_flags)
        *ret_flags = 0;
    if (time_rec)
        *time_rec = 0;
    if (delegated_cred_handle)
        *delegated_cred_handle = GSS_C_NO_CREDENTIAL;

    if (*context_handle == GSS_C_NO_CONTEXT) {
        *minor_status = ENOMEM;
        return GSS_S_FAILURE;
    } else {
        *minor_status = ENOMEM;
        return GSS_S_FAILURE;
    }

    return GSS_S_UNAVAILABLE;
}
