/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
	@header SecSCEP
*/
#include <stdbool.h>

#include <Security/SecKey.h>
#include <Security/SecCertificateRequest.h>
#include <CoreFoundation/CFError.h>

#ifndef _SECURITY_SECSCEP_H_
#define _SECURITY_SECSCEP_H_

__BEGIN_DECLS


SecIdentityRef
SecSCEPCreateTemporaryIdentity(SecKeyRef publicKey, SecKeyRef privateKey);

/*!
    @function SecSCEPGenerateCertificateRequest
    @abstract generate a scep certificate request blob, to be presented to
                a scep server
    @param subject distinguished name to be put in the request
    @param parameters additional information such as challenge and extensions (see SecCMS.h and
         SecCertificateRequest.h for supported keys)
    @param publicKey public key to be certified
    @param privateKey accompanying private key signing the request (proof of possession)
    @param signer identity to sign scep request with, if NULL the keypair to be 
        certified will be turned into a self-signed cert.  The expired identity
        should be passed in case automatic re-enrollment is desired.
    @param recipient SecCertificateRef or CFArray thereof for CA (and optionally RA if used).
*/
CFDataRef
SecSCEPGenerateCertificateRequest(CFArrayRef subject, CFDictionaryRef parameters,
    SecKeyRef publicKey, SecKeyRef privateKey,
    SecIdentityRef signer, CFTypeRef recipient) CF_RETURNS_RETAINED;

/*!
    @function SecSCEPCertifyRequest
    @abstract take a SCEP request and issue a cert
    @param request the request; the ra/ca identity needed to decrypt it needs to be
        in the keychain.
    @param ca_identity to sign the csr
    @param serialno encoded serial number for cert to be issued
	@param pend_request don't issue cert now
*/
CFDataRef
SecSCEPCertifyRequest(CFDataRef request, SecIdentityRef ca_identity, CFDataRef serialno, bool pend_request) CF_RETURNS_RETAINED;

/*!
 @function SecSCEPCertifyRequestWithAlgorithms
 @abstract take a SCEP request and issue a cert
 @param request the request; the ra/ca identity needed to decrypt it needs to be
 in the keychain.
 @param ca_identity to sign the csr
 @param serialno encoded serial number for cert to be issued
 @param pend_request don't issue cert now
 @param hashingAlgorithm hashing algorithm to use, see SecCMS.h
 @param encryptionAlgorithm encryption algorithm to use, see SecCMS.h
 */
CFDataRef
SecSCEPCertifyRequestWithAlgorithms(CFDataRef request, SecIdentityRef ca_identity, CFDataRef serialno, bool pend_request,
                                   CFStringRef hashingAlgorithm, CFStringRef encryptionAlgorithm) CF_RETURNS_RETAINED;

/*!
    @function SecSCEPVerifyReply
    @abstract validate a reply for a sent request and retrieve the issued
        request
    @param request the request sent to the server
    @param reply reply received from server
    @param signer SecCertificateRef or CFArray thereof for CA (and optionally RA if used).
    @param server_error @@@ unused
    @result issued_cert certificate returned in a success reply
*/
CFArrayRef
SecSCEPVerifyReply(CFDataRef request, CFDataRef reply, CFTypeRef signer,
    CFErrorRef *server_error) CF_RETURNS_RETAINED;


/*!
 @function SecSCEPGetCertInitial
 @abstract generate a scep cert initial request, to be presented to
 a scep server, in case the first request timed out
 */
CF_RETURNS_RETAINED
CFDataRef
SecSCEPGetCertInitial(SecCertificateRef ca_certificate, CFArrayRef subject, CFDictionaryRef parameters,
					  CFDictionaryRef signed_attrs, SecIdentityRef signer, CFTypeRef recipient);

/*!
 @function SecSCEPVerifyGetCertInitial
 @abstract Verify signature and encryption on GetCertInitial message
 */
bool
SecSCEPVerifyGetCertInitial(CFDataRef request, SecIdentityRef ca_identity);

/*!
    @function SecSCEPValidateCACertMessage
    @abstract validate GetCACert data against CA fingerprint and find
        appropriate RA certificates if applicable.
    @param certs a PKCS#7 GetCACert response
    @param ca_fingerprint CFDataRef with CA fingerprint.  Size indicates hash type.  Recognises SHA-1 and MD5.
    @param ca_certificate SecCertificateRef CA certificate
    @param ra_signing_certificate SecCertificateRef RA certificate.  Use both for signing and encryption unless ra_encryption_certificate is also returned.
    @param ra_encryption_certificate SecCertificateRef RA encryption certificate.  Returned if there isn't an RA certificate that can both sign and encrypt.
    @result status errSecSuccess on success.
*/
OSStatus 
SecSCEPValidateCACertMessage(CFArrayRef certs,
    CFDataRef ca_fingerprint, SecCertificateRef *ca_certificate, 
    SecCertificateRef *ra_signing_certificate,
    SecCertificateRef *ra_encryption_certificate);


__END_DECLS

#endif /* _SECURITY_SECSCEP_H_ */
