/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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
#include "gssdigest.h"

OM_uint32 _gss_scram_add_cred (
     OM_uint32           *minor_status,
     const gss_cred_id_t input_cred_handle,
     const gss_name_t    desired_name,
     const gss_OID       desired_mech,
     gss_cred_usage_t    cred_usage,
     OM_uint32           initiator_time_req,
     OM_uint32           acceptor_time_req,
     gss_cred_id_t       *output_cred_handle,
     gss_OID_set         *actual_mechs,
     OM_uint32           *initiator_time_rec,
     OM_uint32           *acceptor_time_rec)
{
    if (minor_status)
	*minor_status = 0;
    if (output_cred_handle)
	*output_cred_handle = GSS_C_NO_CREDENTIAL;
    if (actual_mechs)
	*actual_mechs = GSS_C_NO_OID_SET;
    if (initiator_time_rec)
	*initiator_time_rec = 0;
    if (acceptor_time_rec)
	*acceptor_time_rec = 0;
    return GSS_S_COMPLETE;
}
