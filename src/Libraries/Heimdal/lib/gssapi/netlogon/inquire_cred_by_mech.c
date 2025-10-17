/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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

OM_uint32 _netlogon_inquire_cred_by_mech (
            OM_uint32 * minor_status,
            const gss_cred_id_t cred_handle,
            const gss_OID mech_type,
            gss_name_t * name,
            OM_uint32 * initiator_lifetime,
            OM_uint32 * acceptor_lifetime,
            gss_cred_usage_t * cred_usage
    )
{
    OM_uint32 ret;
    const gssnetlogon_cred cred = (const gssnetlogon_cred)cred_handle;

    if (name != NULL) {
        ret = _netlogon_duplicate_name(minor_status,
                                       (const gss_name_t)cred->Name, name);
        if (GSS_ERROR(ret))
            return ret;
    }
    if (initiator_lifetime != NULL)
        *initiator_lifetime = GSS_C_INDEFINITE;
    if (acceptor_lifetime != NULL)
        *acceptor_lifetime = GSS_C_INDEFINITE;
    if (cred_usage != NULL)
        *cred_usage = GSS_C_INITIATE;
    *minor_status = 0;
    return GSS_S_COMPLETE;
}

