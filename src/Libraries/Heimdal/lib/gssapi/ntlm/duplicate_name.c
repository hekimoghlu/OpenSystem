/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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

OM_uint32 _gss_ntlm_duplicate_name (
            OM_uint32 * minor_status,
            const gss_name_t src_name,
            gss_name_t * dest_name
           )
{
    ntlm_name dn = calloc(1, sizeof(*dn));
    ntlm_name sn = (ntlm_name)src_name;

    if (dn) {
	dn->user = strdup(sn->user);
	dn->domain = strdup(sn->domain);
	dn->flags = sn->flags;
	memcpy(dn->ds_uuid, sn->ds_uuid, sizeof(dn->ds_uuid));
	memcpy(dn->uuid, sn->uuid, sizeof(dn->uuid));
    }
    if (dn == NULL || dn->user == NULL || dn->domain == NULL) {
	gss_name_t tempn =  (gss_name_t)dn;
	_gss_ntlm_release_name(minor_status, &tempn);
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    *dest_name = (gss_name_t)dn;
	
    if (minor_status)
	*minor_status = 0;

    return GSS_S_COMPLETE;
}
