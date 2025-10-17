/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#include "krb5_locl.h"
#include "krb5-v4compat.h"

#ifndef HEIMDAL_SMALLER

/**
 * Convert the v5 credentials in in_cred to v4-dito in v4creds.  This
 * is done by sending them to the 524 function in the KDC.  If
 * `in_cred' doesn't contain a DES session key, then a new one is
 * gotten from the KDC and stored in the cred cache `ccache'.
 *
 * @param context Kerberos 5 context.
 * @param in_cred the credential to convert
 * @param v4creds the converted credential
 *
 * @return Returns 0 to indicate success. Otherwise an kerberos et
 * error code is returned, see krb5_get_error_message().
 *
 * @ingroup krb5_v4compat
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb524_convert_creds_kdc(krb5_context context,
			 krb5_creds *in_cred,
			 struct credentials *v4creds)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    memset(v4creds, 0, sizeof(*v4creds));
    krb5_set_error_message(context, EINVAL,
			   N_("krb524_convert_creds_kdc not supported", ""));
    return EINVAL;
}

/**
 * Convert the v5 credentials in in_cred to v4-dito in v4creds,
 * check the credential cache ccache before checking with the KDC.
 *
 * @param context Kerberos 5 context.
 * @param ccache credential cache used to check for des-ticket.
 * @param in_cred the credential to convert
 * @param v4creds the converted credential
 *
 * @return Returns 0 to indicate success. Otherwise an kerberos et
 * error code is returned, see krb5_get_error_message().
 *
 * @ingroup krb5_v4compat
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb524_convert_creds_kdc_ccache(krb5_context context,
				krb5_ccache ccache,
				krb5_creds *in_cred,
				struct credentials *v4creds)
    KRB5_DEPRECATED_FUNCTION("Use X instead")
{
    memset(v4creds, 0, sizeof(*v4creds));
    krb5_set_error_message(context, EINVAL,
			   N_("krb524_convert_creds_kdc_ccache not supported", ""));
    return EINVAL;
}

#endif
