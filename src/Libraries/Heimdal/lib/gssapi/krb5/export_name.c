/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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

static OM_uint32
export_name_common(OM_uint32  *minor_status,
		   gss_const_OID oid,
		   const gss_name_t input_name,
		   gss_buffer_t exported_name)
{
    krb5_const_principal princ = (krb5_const_principal)input_name;
    krb5_error_code ret;
    krb5_context context;
    OM_uint32 major_status;
    char *name;
    
    GSSAPI_KRB5_INIT (&context);
    
    ret = krb5_unparse_name (context, princ, &name);
    if (ret) {
	*minor_status = ret;
	return GSS_S_FAILURE;
    }
    
    major_status = gss_mg_export_name(minor_status, oid, 
				      name, strlen(name),
				      exported_name);
    krb5_xfree(name);
    return major_status;
    
}

OM_uint32
_gsskrb5_export_name(OM_uint32  * minor_status,
		     const gss_name_t input_name,
		     gss_buffer_t exported_name)
{
    return export_name_common(minor_status, GSS_KRB5_MECHANISM, input_name, exported_name);
}

OM_uint32
_gsspku2u_export_name(OM_uint32  * minor_status,
		      const gss_name_t input_name,
		      gss_buffer_t exported_name)
{
    return export_name_common(minor_status, GSS_PKU2U_MECHANISM, input_name, exported_name);
}

OM_uint32
_gssiakerb_export_name(OM_uint32  * minor_status,
		      const gss_name_t input_name,
		      gss_buffer_t exported_name)
{
    return export_name_common(minor_status, GSS_KRB5_MECHANISM, input_name, exported_name);
}
