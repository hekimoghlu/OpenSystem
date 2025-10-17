/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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

OM_uint32 _gss_ntlm_release_cred
           (OM_uint32 * minor_status,
            gss_cred_id_t * cred_handle
           )
{
    gss_name_t name;

    if (minor_status)
	*minor_status = 0;

    if (cred_handle == NULL || *cred_handle == GSS_C_NO_CREDENTIAL)
	return GSS_S_COMPLETE;

    name = (gss_name_t)*cred_handle;
    *cred_handle = GSS_C_NO_CREDENTIAL;

    return _gss_ntlm_release_name(minor_status, &name);
}

