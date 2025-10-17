/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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

OM_uint32
gss_cred_hold(OM_uint32 * __nonnull min_stat, __nonnull gss_cred_id_t cred_handle)
{
    struct _gss_cred *cred = (struct _gss_cred *)cred_handle;
    struct _gss_mechanism_cred *mc;

    *min_stat = 0;

    if (cred == NULL)
	return GSS_S_NO_CRED;

    HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {

	if (mc->gmc_mech->gm_cred_hold == NULL)
	    continue;

	(void)mc->gmc_mech->gm_cred_hold(min_stat, mc->gmc_cred);
    }

    return GSS_S_COMPLETE;
}


OM_uint32
gss_cred_unhold(OM_uint32 * __nonnull min_stat, __nonnull gss_cred_id_t cred_handle)
{
    struct _gss_cred *cred = (struct _gss_cred *)cred_handle;
    struct _gss_mechanism_cred *mc;

    *min_stat = 0;

    if (cred == NULL)
	return GSS_S_NO_CRED;

    HEIM_SLIST_FOREACH(mc, &cred->gc_mc, gmc_link) {

	if (mc->gmc_mech->gm_cred_unhold == NULL)
	    continue;

	(void)mc->gmc_mech->gm_cred_unhold(min_stat, mc->gmc_cred);
    }

    return GSS_S_COMPLETE;
}
