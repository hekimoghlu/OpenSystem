/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
_gsskrb5_pname_to_uid(OM_uint32 *minor_status,
                      const gss_name_t pname,
                      const gss_OID mech_type,
                      uid_t *uidp)
{
#ifdef NO_LOCALNAME
    *minor_status = KRB5_NO_LOCALNAME;
    return GSS_S_FAILURE;
#else
    krb5_error_code ret;
    krb5_context context;
    krb5_const_principal princ = (krb5_const_principal)pname;
    char localname[256];
#ifdef POSIX_GETPWNAM_R
    char pwbuf[2048];
    struct passwd pw, *pwd;
#else
    struct passwd *pwd;
#endif

    GSSAPI_KRB5_INIT(&context);

    *minor_status = 0;

    ret = krb5_aname_to_localname(context, princ,
                                  sizeof(localname), localname);
    if (ret != 0) {
        *minor_status = ret;
        return GSS_S_FAILURE;
    }

#ifdef POSIX_GETPWNAM_R
    if (getpwnam_r(localname, &pw, pwbuf, sizeof(pwbuf), &pwd) != 0) {
        *minor_status = KRB5_NO_LOCALNAME;
        return GSS_S_FAILURE;
    }
#else
    pwd = getpwnam(localname);
#endif

    if (pwd == NULL) {
        *minor_status = KRB5_NO_LOCALNAME;
        return GSS_S_FAILURE;
    }

    *uidp = pwd->pw_uid;

    return GSS_S_COMPLETE;
#endif /* NO_LOCALNAME */
}
