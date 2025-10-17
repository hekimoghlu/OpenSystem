/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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
#if !defined(_OSSL_X509_H_)
#define _OSSL_X509_H_

/*
 * X509 main module
 */
extern VALUE mX509;

/*
 * Converts the VALUE into Integer and set it to the ASN1_TIME. This is a
 * wrapper for X509_time_adj_ex() so passing NULL creates a new ASN1_TIME.
 * Note that the caller must check the NULL return.
 */
ASN1_TIME *ossl_x509_time_adjust(ASN1_TIME *, VALUE);

void Init_ossl_x509(void);

/*
 * X509Attr
 */
extern VALUE cX509Attr;
extern VALUE eX509AttrError;

VALUE ossl_x509attr_new(X509_ATTRIBUTE *);
X509_ATTRIBUTE *GetX509AttrPtr(VALUE);
void Init_ossl_x509attr(void);

/*
 * X509Cert
 */
extern VALUE cX509Cert;
extern VALUE eX509CertError;

VALUE ossl_x509_new(X509 *);
X509 *GetX509CertPtr(VALUE);
X509 *DupX509CertPtr(VALUE);
void Init_ossl_x509cert(void);

/*
 * X509CRL
 */
extern VALUE cX509CRL;
extern VALUE eX509CRLError;

VALUE ossl_x509crl_new(X509_CRL *);
X509_CRL *GetX509CRLPtr(VALUE);
void Init_ossl_x509crl(void);

/*
 * X509Extension
 */
extern VALUE cX509Ext;
extern VALUE cX509ExtFactory;
extern VALUE eX509ExtError;

VALUE ossl_x509ext_new(X509_EXTENSION *);
X509_EXTENSION *GetX509ExtPtr(VALUE);
void Init_ossl_x509ext(void);

/*
 * X509Name
 */
extern VALUE cX509Name;
extern VALUE eX509NameError;

VALUE ossl_x509name_new(X509_NAME *);
X509_NAME *GetX509NamePtr(VALUE);
void Init_ossl_x509name(void);

/*
 * X509Request
 */
extern VALUE cX509Req;
extern VALUE eX509ReqError;

X509_REQ *GetX509ReqPtr(VALUE);
void Init_ossl_x509req(void);

/*
 * X509Revoked
 */
extern VALUE cX509Rev;
extern VALUE eX509RevError;

VALUE ossl_x509revoked_new(X509_REVOKED *);
X509_REVOKED *DupX509RevokedPtr(VALUE);
void Init_ossl_x509revoked(void);

/*
 * X509Store and X509StoreContext
 */
extern VALUE cX509Store;
extern VALUE cX509StoreContext;
extern VALUE eX509StoreError;

X509_STORE *GetX509StorePtr(VALUE);

void Init_ossl_x509store(void);

/*
 * Calls the verify callback Proc (the first parameter) with given pre-verify
 * result and the X509_STORE_CTX.
 */
int ossl_verify_cb_call(VALUE, int, X509_STORE_CTX *);

#endif /* _OSSL_X509_H_ */
