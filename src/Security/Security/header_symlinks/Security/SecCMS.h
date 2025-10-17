/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
	@header SecCMS
*/

#include <CoreFoundation/CoreFoundation.h>
#include <Security/SecIdentity.h>
#include <Security/SecPolicy.h>
#include <Security/SecTrust.h>

#ifndef _SECURITY_SECCMS_H_
#define _SECURITY_SECCMS_H_

__BEGIN_DECLS

extern const void * kSecCMSSignDigest;
extern const void * kSecCMSSignDetached;
extern const void * kSecCMSSignHashAlgorithm;
extern const void * kSecCMSCertChainMode;
extern const void * kSecCMSAdditionalCerts;
extern const void * kSecCMSSignedAttributes;
extern const void * kSecCMSSignDate;
extern const void * kSecCMSAllCerts;
extern const void * kSecCMSHashAgility;
extern const void * kSecCMSHashAgilityV2;
extern const void * kSecCMSExpirationDate;

extern const void * kSecCMSBulkEncryptionAlgorithm;
extern const void * kSecCMSEncryptionAlgorithmDESCBC;
extern const void * kSecCMSEncryptionAlgorithmAESCBC;

extern const void * kSecCMSCertChainModeNone;

extern const void * kSecCMSHashingAlgorithmMD5
    __API_DEPRECATED("Disuse this constant in order to upgrade to SHA-1", ios(3.1, 10.0), macos(10.15, 10.15));
extern const void * kSecCMSHashingAlgorithmSHA1;
extern const void * kSecCMSHashingAlgorithmSHA256;
extern const void * kSecCMSHashingAlgorithmSHA384;
extern const void * kSecCMSHashingAlgorithmSHA512;

/*!
	@function SecCMSVerifyCopyDataAndAttributes
    @abstract verify a signed data cms blob.
    @param message the cms message to be parsed
    @param detached_contents to pass detached contents (optional)
    @param policy specifies policy or array thereof should be used (optional).  
	if none is passed the blob will **not** be verified and only 
	the attached contents will be returned. If verification not desired, please use
    SecCMSDecodeSignedData instead of passing NULL.
    @param trustref (output/optional) if specified, the trust chain built during 
        verification will not be evaluated but returned to the caller to do so.
	@param attached_contents (output/optional) return a copy of the attached 
        contents.
    @param signed_attributes (output/optional) return a copy of the signed
        attributes as a CFDictionary from oids (CFData) to values
        (CFArray of CFData).
    @result A result code.  See "Security Error Codes" (SecBase.h).
        errSecDecode not a CMS message we can parse,
        errSecAuthFailed bad signature, or untrusted signer if caller doesn't
        ask for trustref,
        errSecParam garbage in, garbage out.
*/
OSStatus SecCMSVerifyCopyDataAndAttributes(CFDataRef message, CFDataRef detached_contents,
    CFTypeRef policy, SecTrustRef *trustref,
    CFDataRef *attached_contents, CFDictionaryRef *signed_attributes);

/*!
    @function SecCMSDecodeSignedData
    @abstract decode a signed data cms blob
    @param message the cms message to be parsed
    @param attached_contents (output/optional) return a copy of the attached
        contents.
    @param signed_attributes (output/optional) return a copy of the signed
        attributes as a CFDictionary from oids (CFData) to values
        (CFArray of CFData).
    @result A result code.  See "Security Error Codes" (SecBase.h).
        errSecDecode not a CMS message we can parse,
        errSecAuthFailed bad signature, or untrusted signer if caller doesn't
        ask for trustref,
        errSecParam garbage in, garbage out.
*/
OSStatus SecCMSDecodeSignedData(CFDataRef message,
    CFDataRef *attached_contents, CFDictionaryRef *signed_attributes);

/*!
	@function SecCMSVerify
    @abstract same as SecCMSVerifyCopyDataAndAttributes, for binary compatibility.
*/
OSStatus SecCMSVerify(CFDataRef message, CFDataRef detached_contents,
    CFTypeRef policy, SecTrustRef *trustref, CFDataRef *attached_contents);

OSStatus SecCMSVerifySignedData(CFDataRef message, CFDataRef detached_contents,
    CFTypeRef policy, SecTrustRef *trustref, CFArrayRef additional_certificates,
    CFDataRef *attached_contents, CFDictionaryRef *message_attributes);


/* Return an array of certificates contained in message, if message is of the
   type SignedData and has no signers, return NULL otherwise. */
CFArrayRef SecCMSCertificatesOnlyMessageCopyCertificates(CFDataRef message);

/* Create a degenerate PKCS#7 containing a cert or a CFArray of certs. */
CFDataRef SecCMSCreateCertificatesOnlyMessage(CFTypeRef cert_or_array_thereof);
CFDataRef SecCMSCreateCertificatesOnlyMessageIAP(SecCertificateRef cert);

/*!
	@function SecCMSSignDataAndAttributes
    @abstract create a signed data cms blob.
    @param identity signer
    @param data message to be signed
    @param detached sign detached or not
	@param signed_data (output) return signed message.
    @param signed_attributes (input/optional) signed attributes to insert
        as a CFDictionary from oids (CFData) to value (CFData).
    @result A result code.  See "Security Error Codes" (SecBase.h).
        errSecParam garbage in, garbage out.
*/
OSStatus SecCMSSignDataAndAttributes(SecIdentityRef identity, CFDataRef data, 
    bool detached, CFMutableDataRef signed_data, CFDictionaryRef signed_attributes);

/*!
	@function SecCMSSignDigestAndAttributes
    @abstract create a detached signed data cms blob for a SHA-1 hash.
    @param identity signer
    @param digest SHA-1 digest of message to be signed
	@param signed_data (output) return signed message.
    @param signed_attributes (input/optional) signed attributes to insert
        as a CFDictionary from oids (CFData) to value (CFData).
    @result A result code.  See "Security Error Codes" (SecBase.h).
        errSecParam garbage in, garbage out.
*/
OSStatus SecCMSSignDigestAndAttributes(SecIdentityRef identity, CFDataRef digest, 
    CFMutableDataRef signed_data, CFDictionaryRef signed_attributes);

/*!
	@function SecCMSCreateSignedData
    @abstract create a signed data cms blob.
    @param identity signer
    @param data SHA-1 digest or message to be signed
    @param parameters (input/optional) specify algorithm, detached, digest
    @param signed_attributes (input/optional) signed attributes to insert
        as a CFDictionary from oids (CFData) to value (CFData).
    @param signed_data (output) return signed message.
    @result A result code.  See "Security Error Codes" (SecBase.h).
        errSecParam garbage in, garbage out.
*/
OSStatus SecCMSCreateSignedData(SecIdentityRef identity, CFDataRef data, 
    CFDictionaryRef parameters, CFDictionaryRef signed_attributes,
    CFMutableDataRef signed_data);

/*!
    @function SecCMSCreateEnvelopedData
    @abstract create a enveloped cms blob for recipients
    @param recipient_or_cfarray_thereof SecCertificateRef for each recipient
    @param params CFDictionaryRef with encryption parameters
    @param data Data to be encrypted
    @param enveloped_data (output) return enveloped message.
    @result A result code.  See "Security Error Codes" (SecBase.h).
        errSecParam garbage in, garbage out.
*/
OSStatus SecCMSCreateEnvelopedData(CFTypeRef recipient_or_cfarray_thereof, 
    CFDictionaryRef params, CFDataRef data, CFMutableDataRef enveloped_data);

/*!
    @function SecCMSDecryptEnvelopedData
    @abstract open an enveloped cms blob. expects recipients identity in keychain.
    @param message Eveloped message
    @param data (output) return decrypted message.
    @param recipient (output/optional) return addressed recipient
    @result A result code.  See "Security Error Codes" (SecBase.h).
        errSecParam garbage in, garbage out.
*/
OSStatus SecCMSDecryptEnvelopedData(CFDataRef message, 
    CFMutableDataRef data, SecCertificateRef *recipient);

bool useMessageSecurityEnabled(void);

__END_DECLS

#endif /* !_SECURITY_SECCMS_H_ */
