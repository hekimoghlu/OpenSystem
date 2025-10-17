/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include "ftpd_locl.h"
#include <gssapi/gssapi.h>

/* XXX sync with gssapi.c */
struct gssapi_data {
    gss_ctx_id_t context_hdl;
    gss_name_t client_name;
    gss_cred_id_t delegated_cred_handle;
    void *mech_data;
};

int gssapi_userok(void*, char*); /* to keep gcc happy */
int gssapi_session(void*, char*); /* to keep gcc happy */

int
gssapi_userok(void *app_data, char *username)
{
    struct gssapi_data *data = app_data;

    /* Yes, this logic really is inverted. */
    return !gss_userok(data->client_name, username);
}

int
gssapi_session(void *app_data, char *username)
{
    struct gssapi_data *data = app_data;
    OM_uint32 major, minor;
    int ret = 0;

    if (data->delegated_cred_handle != GSS_C_NO_CREDENTIAL) {
        major = gss_store_cred(&minor, data->delegated_cred_handle,
                               GSS_C_INITIATE, GSS_C_NO_OID,
                               1, 1, NULL, NULL);
        if (GSS_ERROR(major))
            ret = 1;
	afslog(NULL, 1);
    }

    gss_release_cred(&minor, &data->delegated_cred_handle);
    return ret;
}
