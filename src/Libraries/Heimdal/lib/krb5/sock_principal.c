/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_sock_to_principal (krb5_context context,
			int sock,
			const char *sname,
			int32_t type,
			krb5_principal *ret_princ)
{
    krb5_error_code ret;
    struct sockaddr_storage __ss;
    struct sockaddr *sa = (struct sockaddr *)&__ss;
    socklen_t salen = sizeof(__ss);
    char hostname[NI_MAXHOST];

    if (getsockname (sock, sa, &salen) < 0) {
	ret = errno;
	krb5_set_error_message (context, ret, "getsockname: %s", strerror(ret));
	return ret;
    }
    ret = getnameinfo (sa, salen, hostname, sizeof(hostname), NULL, 0, 0);
    if (ret) {
	int save_errno = errno;
	krb5_error_code ret2 = krb5_eai_to_heim_errno(ret, save_errno);
	krb5_set_error_message (context, ret2, "getnameinfo: %s", gai_strerror(ret));
	return ret2;
    }

    ret = krb5_sname_to_principal (context,
				   hostname,
				   sname,
				   type,
				   ret_princ);
    return ret;
}
