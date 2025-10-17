/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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

/**
 * Release a gss_OID
 *
 * This function should never be used, this is since many of the
 * gss_OID objects passed around are stack and data objected that are
 * not free-able.
 *
 * The function tries to find internal OIDs that are static and avoid
 * trying to free them.
 *
 * One could guess that gss_name_to_oid() might return an allocated
 * OID.  In this implementation it wont, so there is no need to call
 * gss_release_oid().
 *
 * @param minor_status minor status code returned
 * @param oid oid to be released/freed.
 *
 * @returns GSS major status code
 *
 * @ingroup gssapi
 */

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_release_oid(OM_uint32 *__nonnull minor_status,
		__nullable gss_OID * __nonnull oid)
{
    struct _gss_mech_switch *m;
    gss_OID o = *oid;
    size_t n;

    *oid = GSS_C_NO_OID;

    if (minor_status != NULL)
	*minor_status = 0;

    if (o == GSS_C_NO_OID)
	return GSS_S_COMPLETE;

    /*
     * Program broken and tries to release an static oid, don't let
     * it crash us, search static tables forst.
     */
    for (n = 0; _gss_ont_mech[n].oid; n++)
	if (_gss_ont_mech[n].oid == o)
	    return GSS_S_COMPLETE;
    for (n = 0; _gss_ont_ma[n].oid; n++)
	if (_gss_ont_ma[n].oid == o)
	    return GSS_S_COMPLETE;

    HEIM_SLIST_FOREACH(m, &_gss_mechs, gm_link) {
	if (&m->gm_mech.gm_mech_oid == o)
	    return GSS_S_COMPLETE;
    }

    /* ok, the program doesn't try to crash us, lets release it for real now */

    if (o->elements != NULL) {
	free(o->elements);
	o->elements = NULL;
    }
    o->length = 0;
    free(o);

    return GSS_S_COMPLETE;
}
