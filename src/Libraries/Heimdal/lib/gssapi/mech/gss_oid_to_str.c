/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 28, 2024.
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
 * Turn an mech OID into an name
 *
 * Try to turn a OID into a mechanism name. If a matching OID can't be
 * found, this function will return NULL.
 *
 * The caller must free the oid_str buffer with gss_release_buffer()
 * when done with the string.
 *	  
 * @param minor_status an minor status code
 * @param oid an oid
 * @param oid_str buffer that will point to a NUL terminated string that is the numreric OID
 *
 * @returns a gss major status code
 *
 * @ingroup gssapi
 */

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_oid_to_str(OM_uint32 *__nonnull minor_status,
	       __nonnull gss_OID oid,
	       __nonnull gss_buffer_t oid_str)
{
    int ret;
    size_t size;
    heim_oid o;
    char *p;

    _mg_buffer_zero(oid_str);

    if (oid == GSS_C_NULL_OID)
	return GSS_S_FAILURE;

    ret = der_get_oid (oid->elements, oid->length, &o, &size);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }

    ret = der_print_heim_oid(&o, ' ', &p);
    der_free_oid(&o);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }

    oid_str->value = p;
    oid_str->length = strlen(p);

    *minor_status = 0;
    return GSS_S_COMPLETE;
}

/**
 * Turn an mech OID into an name
 *
 * Try to turn a OID into a mechanism name. If a matching OID can't be
 * found, this function will return NULL.
 *
 * @param oid an mechanism oid
 *
 * @returns pointer a static strng to a name or NULL when not found.
 *	  
 * @ingroup gssapi
 */

GSSAPI_LIB_FUNCTION const char * __nullable GSSAPI_LIB_CALL
gss_oid_to_name(__nonnull gss_const_OID oid)
{
    size_t i;

    for (i = 0; _gss_ont_mech[i].oid; i++) {
	if (gss_oid_equal(oid, _gss_ont_mech[i].oid)) {
	    if (_gss_ont_mech[i].short_desc)
		return _gss_ont_mech[i].short_desc;
	    return _gss_ont_mech[i].name;
	}
    }
    return NULL;
}

/**
 * Turn an mech name into an OID
 *
 * Try to turn a string/name into an OID. If a static OID can't be
 * found, this function will return NULL.
 * 
 * WARNING: do _NOT_ call gss_release_oid() on the OID returned.
 *
 * @param name name of a OID
 *
 * @returns pointer a static OID or GSS_C_NO_OID when not found.
 *
 *	  
 * @ingroup gssapi
 */

GSSAPI_LIB_FUNCTION __nullable gss_const_OID GSSAPI_LIB_CALL
gss_name_to_oid(const char *__nonnull name)
{
    size_t i, partial = (size_t)-1;

    for (i = 0; _gss_ont_mech[i].oid; i++) {
	if (strcasecmp(name, _gss_ont_mech[i].short_desc) == 0)
	    return _gss_ont_mech[i].oid;
	if (strncasecmp(name, _gss_ont_mech[i].short_desc, strlen(name)) == 0) {
	    if (partial != (size_t)-1)
		return NULL;
	    partial = i;
	}
    }
    if (partial != (size_t)-1)
	return _gss_ont_mech[partial].oid;

    return NULL;
}
