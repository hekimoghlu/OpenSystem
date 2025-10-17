/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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

OM_uint32 GSSAPI_CALLCONV _gsskrb5_display_name
           (OM_uint32 * minor_status,
            const gss_name_t input_name,
            gss_buffer_t output_name_buffer,
            gss_OID * output_name_type
           )
{
    krb5_const_principal name = (krb5_const_principal)input_name;
    krb5_context context;
    krb5_error_code kret;
    char *str;

    GSSAPI_KRB5_INIT (&context);

    kret = krb5_unparse_name_flags(context, name,
				   KRB5_PRINCIPAL_UNPARSE_DISPLAY, &str);
    if (kret) {
	*minor_status = kret;
	return GSS_S_FAILURE;
    }

    output_name_buffer->length = strlen(str);
    output_name_buffer->value  = str;

    if (output_name_type)
	*output_name_type = GSS_KRB5_NT_PRINCIPAL_NAME;
    *minor_status = 0;
    return GSS_S_COMPLETE;
}
