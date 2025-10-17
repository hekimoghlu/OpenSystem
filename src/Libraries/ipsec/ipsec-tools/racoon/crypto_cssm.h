/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 6, 2023.
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
#ifndef __CRYPTO_CSSM_H__
#define __CRYPTO_CSSM_H__

/*
 * Racoon module for verifying and signing certificates through Security
 * Framework and CSSM
 */

#include "vmbuf.h"
#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecBase.h>


extern cert_status_t crypto_cssm_check_x509cert_dates (SecCertificateRef certificateRef);
extern int crypto_cssm_check_x509cert (cert_t *hostcert, cert_t *certchain, CFStringRef hostname, SecKeyRef *publicKeyRef);
extern int crypto_cssm_verify_x509sign(SecKeyRef publicKeyRef, vchar_t *hash, vchar_t *signature, Boolean useSHA1);
extern SecCertificateRef crypto_cssm_x509cert_CreateSecCertificateRef (vchar_t *cert);
extern vchar_t* crypto_cssm_getsign(CFDataRef persistentCertRef, vchar_t* hash);
extern vchar_t* crypto_cssm_get_x509cert(CFDataRef persistentCertRef, cert_status_t *certStatus);
extern const char *GetSecurityErrorString(OSStatus err);
extern CFDataRef crypto_cssm_CopySubjectSequence(SecCertificateRef certRef);

#endif /* __CRYPTO_CSSM_H__ */

