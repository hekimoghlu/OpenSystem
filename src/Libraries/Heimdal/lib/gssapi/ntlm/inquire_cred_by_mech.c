/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#include "ntlm.h"

OM_uint32 _gss_ntlm_inquire_cred_by_mech (
            OM_uint32 * minor_status,
            const gss_cred_id_t cred_handle,
            const gss_OID mech_type,
            gss_name_t * name,
            OM_uint32 * initiator_lifetime,
            OM_uint32 * acceptor_lifetime,
            gss_cred_usage_t * cred_usage
    )
{
    if (name) {
	if (cred_handle) {
	    OM_uint32 major_status;
	    major_status = _gss_ntlm_duplicate_name(minor_status, (gss_name_t)cred_handle, (gss_name_t *)name);
	    if (major_status != GSS_S_COMPLETE)
		return major_status;
	} else
	    *name = GSS_C_NO_NAME;
    }
    if (initiator_lifetime)
	*initiator_lifetime = GSS_C_INDEFINITE;
    if (acceptor_lifetime)
	*acceptor_lifetime = 0;
    if (cred_usage)
	*cred_usage = GSS_C_INITIATE;
    if (minor_status)
	*minor_status = 0;

    return GSS_S_COMPLETE;
}
