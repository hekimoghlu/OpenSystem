/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
/*!
    @header SecCmsRecipientInfo.h

    @availability 10.4 and later
    @abstract Interfaces of the CMS implementation.
    @discussion The functions here implement functions for encoding
                and decoding Cryptographic Message Syntax (CMS) objects
                as described in rfc3369.
 */

#ifndef _SECURITY_SECCMSRECIPIENTINFO_H_
#define _SECURITY_SECCMSRECIPIENTINFO_H_  1

#include <Security/SecCmsBase.h>

__BEGIN_DECLS

#if TARGET_OS_OSX
/*!
    @function
    @abstract Create a recipientinfo
    @discussion We currently do not create KeyAgreement recipientinfos with multiple recipientEncryptedKeys
    the certificate is supposed to have been verified by the caller
 */
extern SecCmsRecipientInfoRef
SecCmsRecipientInfoCreate(SecCmsMessageRef cmsg, SecCertificateRef cert)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);

#else // !TARGET_OS_OSX

/*!
    @function
    @abstract Create a recipientinfo
    @discussion We currently do not create KeyAgreement recipientinfos with multiple recipientEncryptedKeys
                the certificate is supposed to have been verified by the caller
 */
extern SecCmsRecipientInfoRef
SecCmsRecipientInfoCreate(SecCmsEnvelopedDataRef envd, SecCertificateRef cert)
    API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#endif // !TARGET_OS_OSX


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#if TARGET_OS_OSX
extern SecCmsRecipientInfoRef
SecCmsRecipientInfoCreateWithSubjKeyID(SecCmsMessageRef cmsg,
                                       CSSM_DATA_PTR subjKeyID,
                                       SecPublicKeyRef pubKey)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);
#else // !TARGET_OS_OSX
extern SecCmsRecipientInfoRef
SecCmsRecipientInfoCreateWithSubjKeyID(SecCmsEnvelopedDataRef envd,
                                       const SecAsn1Item *subjKeyID,
                                       SecPublicKeyRef pubKey)
    API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#endif // !TARGET_OS_OSX
#pragma clang diagnostic pop

#if TARGET_OS_OSX
extern SecCmsRecipientInfoRef
SecCmsRecipientInfoCreateWithSubjKeyIDFromCert(SecCmsMessageRef cmsg,
                                               SecCertificateRef cert)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);
#else // !TARGET_OS_OSX
extern SecCmsRecipientInfoRef
SecCmsRecipientInfoCreateWithSubjKeyIDFromCert(SecCmsEnvelopedDataRef envd, 
                                               SecCertificateRef cert)
    API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#endif // !TARGET_OS_OSX


#if TARGET_OS_OSX
extern void
SecCmsRecipientInfoDestroy(SecCmsRecipientInfoRef ri);
#endif

__END_DECLS

#endif /* _SECURITY_SECCMSRECIPIENTINFO_H_ */
