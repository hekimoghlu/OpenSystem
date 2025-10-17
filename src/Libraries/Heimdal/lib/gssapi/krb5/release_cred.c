/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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

OM_uint32 GSSAPI_CALLCONV _gsskrb5_release_cred
           (OM_uint32 * minor_status,
            gss_cred_id_t * cred_handle
           )
{
    krb5_context context;
    gsskrb5_cred cred;

    *minor_status = 0;

    if (*cred_handle == NULL)
        return GSS_S_COMPLETE;

    cred = (gsskrb5_cred)*cred_handle;
    *cred_handle = GSS_C_NO_CREDENTIAL;

    GSSAPI_KRB5_INIT (&context);

    HEIMDAL_MUTEX_lock(&cred->cred_id_mutex);

    if (cred->principal != NULL)
        krb5_free_principal(context, cred->principal);
    if (cred->keytab != NULL)
	krb5_kt_close(context, cred->keytab);
    if (cred->ccache != NULL) {
	if (cred->cred_flags & GSS_CF_DESTROY_CRED_ON_RELEASE)
	    krb5_cc_destroy(context, cred->ccache);
	else
	    krb5_cc_close(context, cred->ccache);
    }
    if (cred->enctypes)
	free(cred->enctypes);
#ifdef PKINIT
    if (cred->cert)
	hx509_cert_free(cred->cert);
#endif
    if (cred->password) {
	memset(cred->password, 0, strlen(cred->password));
	free(cred->password);
    }
    HEIMDAL_MUTEX_unlock(&cred->cred_id_mutex);
    HEIMDAL_MUTEX_destroy(&cred->cred_id_mutex);
    memset(cred, 0, sizeof(*cred));
    free(cred);
    return GSS_S_COMPLETE;
}

