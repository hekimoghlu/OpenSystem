/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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

//
//  SecCMSInternal.h
//  Security
//
//  WARNING: This header contains the shim functions for SecCMS using MessageSecurity.
//  It will be removed when the legacy implementations are removed.

#ifndef _SECURITY_SECCMS_INTERNAL_H_
#define _SECURITY_SECCMS_INTERNAL_H_

#include <Security/SecCMS.h>

__BEGIN_DECLS

/* Return an array of certificates contained in message, if message is of the
   type SignedData and has no signers, return NULL otherwise. */
CF_RETURNS_RETAINED CFArrayRef
MS_SecCMSCertificatesOnlyMessageCopyCertificates(CFDataRef message);

OSStatus MS_SecCMSVerifySignedData_internal(CFDataRef message, CFDataRef detached_contents,
                                            CFTypeRef policy, SecTrustRef CF_RETURNS_RETAINED *trustref, CFArrayRef additional_certs,
                                            CFDataRef CF_RETURNS_RETAINED *attached_contents, CFDictionaryRef CF_RETURNS_RETAINED *signed_attributes);

OSStatus MS_SecCMSDecodeSignedData(CFDataRef message,
                                   CFDataRef *attached_contents, CFDictionaryRef *signed_attributes);

OSStatus MS_SecCMSDecryptEnvelopedData(CFDataRef message, CFMutableDataRef data,
                                       SecCertificateRef *recipient);

__END_DECLS

#endif /* _SECURITY_SECCMS_INTERNAL_H_ */
