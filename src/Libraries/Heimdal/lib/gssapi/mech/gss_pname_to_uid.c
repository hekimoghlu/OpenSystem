/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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

static OM_uint32
mech_pname_to_uid(OM_uint32 *minor_status,
                  struct _gss_mechanism_name *mn,
                  uid_t *uidp)
{
    OM_uint32 major_status = GSS_S_UNAVAILABLE;

    *minor_status = 0;

    if (mn->gmn_mech->gm_pname_to_uid == NULL)
        return GSS_S_UNAVAILABLE;

    major_status = mn->gmn_mech->gm_pname_to_uid(minor_status,
                                                 mn->gmn_name,
                                                 mn->gmn_mech_oid,
                                                 uidp);
    if (GSS_ERROR(major_status))
        _gss_mg_error(mn->gmn_mech, *minor_status);

    return major_status;
}

static OM_uint32
attr_pname_to_uid(OM_uint32 *minor_status,
                  struct _gss_mechanism_name *mn,
                  uid_t *uidp)
{
#ifdef NO_LOCALNAME
    return GSS_S_UNAVAILABLE;
#else
    OM_uint32 major_status = GSS_S_UNAVAILABLE;
    OM_uint32 tmpMinor;
    int more = -1;

    *minor_status = 0;

    if (mn->gmn_mech->gm_get_name_attribute == NULL)
        return GSS_S_UNAVAILABLE;

    while (more != 0) {
        gss_buffer_desc value;
        gss_buffer_desc display_value;
        int authenticated = 0, complete = 0;
#ifdef POSIX_GETPWNAM_R
        char pwbuf[2048];
        struct passwd pw, *pwd;
#else
        struct passwd *pwd;
#endif
        char *localname;

        major_status = mn->gmn_mech->gm_get_name_attribute(minor_status,
                                                           mn->gmn_name,
                                                           GSS_C_ATTR_LOCAL_LOGIN_USER,
                                                           &authenticated,
                                                           &complete,
                                                           &value,
                                                           &display_value,
                                                           &more);
        if (GSS_ERROR(major_status)) {
            _gss_mg_error(mn->gmn_mech, *minor_status);
            break;
        }

        localname = malloc(value.length + 1);
        if (localname == NULL) {
            major_status = GSS_S_FAILURE;
            *minor_status = ENOMEM;
            break;
        }

        memcpy(localname, value.value, value.length);
        localname[value.length] = '\0';

#ifdef POSIX_GETPWNAM_R
        if (getpwnam_r(localname, &pw, pwbuf, sizeof(pwbuf), &pwd) != 0)
            pwd = NULL;
#else
        pwd = getpwnam(localname);
#endif

        free(localname);
        gss_release_buffer(&tmpMinor, &value);
        gss_release_buffer(&tmpMinor, &display_value);

        if (pwd != NULL) {
            *uidp = pwd->pw_uid;
            major_status = GSS_S_COMPLETE;
            *minor_status = 0;
            break;
        } else
            major_status = GSS_S_UNAVAILABLE;
    }

    return major_status;
#endif /* NO_LOCALNAME */
}

GSSAPI_LIB_FUNCTION OM_uint32 GSSAPI_LIB_CALL
gss_pname_to_uid(OM_uint32 *__nonnull minor_status,
                 __nonnull const gss_name_t pname,
                 __nonnull const gss_OID mech_type,
                 uid_t *__nonnull uidp)
{
    OM_uint32 major_status = GSS_S_UNAVAILABLE;
    struct _gss_name *name = (struct _gss_name *) pname;
    struct _gss_mechanism_name *mn = NULL;

    *minor_status = 0;

    if (mech_type != GSS_C_NO_OID) {
        major_status = _gss_find_mn(minor_status, name, mech_type, &mn);
        if (GSS_ERROR(major_status))
            return major_status;

        major_status = mech_pname_to_uid(minor_status, mn, uidp);
        if (major_status != GSS_S_COMPLETE)
            major_status = attr_pname_to_uid(minor_status, mn, uidp);
    } else {
        HEIM_SLIST_FOREACH(mn, &name->gn_mn, gmn_link) {
            major_status = mech_pname_to_uid(minor_status, mn, uidp);
            if (major_status != GSS_S_COMPLETE)
                major_status = attr_pname_to_uid(minor_status, mn, uidp);
            if (major_status != GSS_S_UNAVAILABLE)
                break;
        }
    }

    if (major_status != GSS_S_COMPLETE && mn != NULL)
        _gss_mg_error(mn->gmn_mech, *minor_status);

    return major_status;
}
