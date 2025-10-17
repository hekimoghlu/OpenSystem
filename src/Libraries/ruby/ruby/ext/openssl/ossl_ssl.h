/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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
/*
 * This program is licensed under the same licence as Ruby.
 * (See the file 'LICENCE'.)
 */
#if !defined(_OSSL_SSL_H_)
#define _OSSL_SSL_H_

#define GetSSL(obj, ssl) do { \
	TypedData_Get_Struct((obj), SSL, &ossl_ssl_type, (ssl)); \
	if (!(ssl)) { \
		ossl_raise(rb_eRuntimeError, "SSL is not initialized"); \
	} \
} while (0)

#define GetSSLSession(obj, sess) do { \
	TypedData_Get_Struct((obj), SSL_SESSION, &ossl_ssl_session_type, (sess)); \
	if (!(sess)) { \
		ossl_raise(rb_eRuntimeError, "SSL Session wasn't initialized."); \
	} \
} while (0)

extern const rb_data_type_t ossl_ssl_type;
extern const rb_data_type_t ossl_ssl_session_type;
extern VALUE mSSL;
extern VALUE cSSLSocket;
extern VALUE cSSLSession;

void Init_ossl_ssl(void);
void Init_ossl_ssl_session(void);

#endif /* _OSSL_SSL_H_ */
