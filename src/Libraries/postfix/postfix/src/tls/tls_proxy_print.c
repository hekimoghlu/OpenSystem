/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 5, 2021.
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
#ifdef USE_TLS

/* System library. */

#include <sys_defs.h>

/* Utility library */

#include <attr.h>

/* Global library. */

#include <mail_proto.h>

/* TLS library. */

#include <tls.h>
#include <tls_proxy.h>

/* tls_proxy_context_print - send TLS session state over stream */

int     tls_proxy_context_print(ATTR_PRINT_MASTER_FN print_fn, VSTREAM *fp,
				        int flags, void *ptr)
{
    TLS_SESS_STATE *tp = (TLS_SESS_STATE *) ptr;
    int     ret;

#define STRING_OR_EMPTY(s) ((s) ? (s) : "")

    ret = print_fn(fp, flags | ATTR_FLAG_MORE,
		   SEND_ATTR_STR(MAIL_ATTR_PEER_CN,
				 STRING_OR_EMPTY(tp->peer_CN)),
		   SEND_ATTR_STR(MAIL_ATTR_ISSUER_CN,
				 STRING_OR_EMPTY(tp->issuer_CN)),
		   SEND_ATTR_STR(MAIL_ATTR_PEER_CERT_FPT,
				 STRING_OR_EMPTY(tp->peer_cert_fprint)),
		   SEND_ATTR_STR(MAIL_ATTR_PEER_PKEY_FPT,
				 STRING_OR_EMPTY(tp->peer_pkey_fprint)),
		   SEND_ATTR_INT(MAIL_ATTR_PEER_STATUS,
				 tp->peer_status),
		   SEND_ATTR_STR(MAIL_ATTR_CIPHER_PROTOCOL,
				 STRING_OR_EMPTY(tp->protocol)),
		   SEND_ATTR_STR(MAIL_ATTR_CIPHER_NAME,
				 STRING_OR_EMPTY(tp->cipher_name)),
		   SEND_ATTR_INT(MAIL_ATTR_CIPHER_USEBITS,
				 tp->cipher_usebits),
		   SEND_ATTR_INT(MAIL_ATTR_CIPHER_ALGBITS,
				 tp->cipher_algbits),
		   ATTR_TYPE_END);
    return (ret);
}

#endif
