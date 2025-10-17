/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
#include <krb5_locl.h>

/* These are stub functions for the standalone RFC3961 crypto library */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_init_context(krb5_context *context)
{
    krb5_context p;

    *context = NULL;

    /* should have a run_once */
    bindtextdomain(HEIMDAL_TEXTDOMAIN, HEIMDAL_LOCALEDIR);

    p = calloc(1, sizeof(*p));
    if(!p)
        return ENOMEM;

    p->mutex = malloc(sizeof(HEIMDAL_MUTEX));
    if (p->mutex == NULL) {
        free(p);
        return ENOMEM;
    }
    HEIMDAL_MUTEX_init(p->mutex);

    *context = p;
    return 0;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_free_context(krb5_context context)
{
    krb5_clear_error_message(context);

    HEIMDAL_MUTEX_destroy(context->mutex);
    free(context->mutex);
    if (context->flags & KRB5_CTX_F_SOCKETS_INITIALIZED) {
        rk_SOCK_EXIT();
    }

    memset(context, 0, sizeof(*context));
    free(context);
}

krb5_boolean
krb5_homedir_access(krb5_context context)
{
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_log(krb5_context context,
         krb5_log_facility *fac,
         int level,
         const char *fmt,
         ...)
{
    return 0;
}

/* This function is currently just used to get the location of the EGD
 * socket. If we're not using an EGD, then we can just return NULL */

KRB5_LIB_FUNCTION const char* KRB5_LIB_CALL
krb5_config_get_string (krb5_context context,
                        const krb5_config_section *c,
                        ...)
{
    return NULL;
}
