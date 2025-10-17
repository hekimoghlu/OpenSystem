/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
gss_seal(OM_uint32 *__nonnull minor_status,
    __nonnull gss_ctx_id_t context_handle,
    int conf_req_flag,
    int qop_req,
    __nonnull gss_buffer_t input_message_buffer,
    int *__nonnull conf_state,
    __nonnull gss_buffer_t output_message_buffer)
{

	return (gss_wrap(minor_status,
		    context_handle, conf_req_flag, qop_req,
		    input_message_buffer, conf_state,
		    output_message_buffer));
}
