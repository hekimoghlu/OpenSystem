/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 30, 2025.
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

OM_uint32 GSSAPI_CALLCONV _gsskrb5_inquire_cred_by_mech (
    OM_uint32 * minor_status,
	const gss_cred_id_t cred_handle,
	const gss_OID mech_type,
	gss_name_t * name,
	OM_uint32 * initiator_lifetime,
	OM_uint32 * acceptor_lifetime,
	gss_cred_usage_t * cred_usage
    )
{
    gss_cred_usage_t usage;
    OM_uint32 maj_stat;
    OM_uint32 lifetime;

    maj_stat =
	_gsskrb5_inquire_cred (minor_status, cred_handle,
			       name, &lifetime, &usage, NULL);
    if (maj_stat)
	return maj_stat;

    if (initiator_lifetime) {
	if (usage == GSS_C_INITIATE || usage == GSS_C_BOTH)
	    *initiator_lifetime = lifetime;
	else
	    *initiator_lifetime = 0;
    }

    if (acceptor_lifetime) {
	if (usage == GSS_C_ACCEPT || usage == GSS_C_BOTH)
	    *acceptor_lifetime = lifetime;
	else
	    *acceptor_lifetime = 0;
    }

    if (cred_usage)
	*cred_usage = usage;

    return GSS_S_COMPLETE;
}
