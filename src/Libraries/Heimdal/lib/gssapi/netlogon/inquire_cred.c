/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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

OM_uint32 _netlogon_inquire_cred
           (OM_uint32 * minor_status,
            const gss_cred_id_t cred_handle,
            gss_name_t * name,
            OM_uint32 * lifetime,
            gss_cred_usage_t * cred_usage,
            gss_OID_set * mechanisms
           )
{
    OM_uint32 ret;
    const gssnetlogon_cred cred = (const gssnetlogon_cred)cred_handle;

    *minor_status = 0;

    if (cred == NULL)
	return GSS_S_NO_CRED;

    if (name != NULL) {
        ret = _netlogon_duplicate_name(minor_status,
                                       (const gss_name_t)cred->Name, name);
        if (GSS_ERROR(ret))
            return ret;
    }
    if (lifetime != NULL)
        *lifetime = GSS_C_INDEFINITE;
    if (cred_usage != NULL)
        *cred_usage = GSS_C_INITIATE;
    if (mechanisms != NULL)
        *mechanisms = GSS_C_NO_OID_SET;
    return GSS_S_COMPLETE;
}
