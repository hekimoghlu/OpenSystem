/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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

OM_uint32 GSSAPI_CALLCONV
_gsskrb5_authorize_localname(OM_uint32 *minor_status,
                             const gss_name_t input_name,
                             gss_const_buffer_t user_name,
                             gss_const_OID user_name_type)
{
    krb5_context context;
    krb5_principal princ = (krb5_principal)input_name;
    char *user;
    int user_ok;

    if (!gss_oid_equal(user_name_type, GSS_C_NT_USER_NAME))
        return GSS_S_BAD_NAMETYPE;

    GSSAPI_KRB5_INIT(&context);

    user = malloc(user_name->length + 1);
    if (user == NULL) {
        *minor_status = ENOMEM;
        return GSS_S_FAILURE;
    }

    memcpy(user, user_name->value, user_name->length);
    user[user_name->length] = '\0';

    *minor_status = 0;
    user_ok = krb5_kuserok(context, princ, user);

    free(user);

    return user_ok ? GSS_S_COMPLETE : GSS_S_UNAUTHORIZED;
}
