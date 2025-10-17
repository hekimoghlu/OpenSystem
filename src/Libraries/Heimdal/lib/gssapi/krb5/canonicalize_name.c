/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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

OM_uint32 GSSAPI_CALLCONV _gsskrb5_canonicalize_name (
            OM_uint32 * minor_status,
            const gss_name_t input_name,
            const gss_OID mech_type,
            gss_name_t * output_name
           )
{
    krb5_context context;
    krb5_principal name;
    OM_uint32 ret;

    *output_name = NULL;

    GSSAPI_KRB5_INIT (&context);

    ret = _gsskrb5_canon_name(minor_status, context, 1, NULL, input_name, &name);
    if (ret)
	return ret;

    *output_name = (gss_name_t)name;

    return GSS_S_COMPLETE;
}
