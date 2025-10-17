/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_export_name(OM_uint32 *__nonnull minor_status,
    __nonnull const gss_name_t input_name,
    __nonnull gss_buffer_t exported_name)
{
	struct _gss_name *name = (struct _gss_name *) input_name;
	struct _gss_mechanism_name *mn;

	_gss_mg_check_name(input_name);
	_mg_buffer_zero(exported_name);

	/*
	 * If this name already has any attached MNs, export the first
	 * one, otherwise export based on the first mechanism in our
	 * list.
	 */
	mn = HEIM_SLIST_FIRST(&name->gn_mn);
	if (!mn) {
		*minor_status = 0;
		return (GSS_S_NAME_NOT_MN);
	}

	return mn->gmn_mech->gm_export_name(minor_status,
	    mn->gmn_name, exported_name);
}

OM_uint32
gss_mg_export_name(OM_uint32 * __nonnull minor_status,
		   __nonnull const gss_const_OID mech,
		    const void *__nonnull name,
		   size_t length, 
		   __nonnull gss_buffer_t exported_name)
{
    uint8_t *buf;

    exported_name->length = 10 + length + mech->length;
    exported_name->value  = malloc(exported_name->length);
    if (exported_name->value == NULL) {
	*minor_status = ENOMEM;
	return GSS_S_FAILURE;
    }

    /* TOK, MECH_OID_LEN, DER(MECH_OID), NAME_LEN, NAME */

    buf = exported_name->value;
    memcpy(buf, "\x04\x01", 2);
    buf += 2;
    buf[0] = ((mech->length + 2) >> 8) & 0xff;
    buf[1] = (mech->length + 2) & 0xff;
    buf+= 2;
    buf[0] = 0x06;
    buf[1] = (mech->length) & 0xFF;
    buf+= 2;

    memcpy(buf, mech->elements, mech->length);
    buf += mech->length;

    buf[0] = (length >> 24) & 0xff;
    buf[1] = (length >> 16) & 0xff;
    buf[2] = (length >> 8) & 0xff;
    buf[3] = (length) & 0xff;
    buf += 4;

    memcpy (buf, name, length);

    *minor_status = 0;
    return GSS_S_COMPLETE;
}
