/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
 @header SecCodeSignerRemote
 SecCodeSignerRemote represents an object that can sign code without using local identities and keys, instead
 requiring the caller to provide the signing implementation.
 */
#ifndef _H_SECCODESIGNERREMOTE
#define _H_SECCODESIGNERREMOTE

#include <Security/CSCommon.h>
#include <Security/SecCodeSigner.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 @typedef SecCodeSignerRemoteRef
 This is the type of a reference to a code signer meant to leverage remote signing keys.
 */
typedef struct CF_BRIDGED_TYPE(id) __SecCodeSignerRemote *SecCodeSignerRemoteRef SPI_AVAILABLE(macos(13.3));

/*!
 @function SecCodeSignerRemoteGetTypeID
 Returns the type identifier of all SecCodeSignerRemote instances.
 */
CFTypeID SecCodeSignerRemoteGetTypeID(void) SPI_AVAILABLE(macos(13.3));

/*!
 @function SecCodeSignerRemoteCreate
 Create a (new) SecCodeSignerRemote object to be used for signing code with a remote key.

 @param parameters An optional CFDictionary containing parameters that influence
 signing operations with the newly created SecCodeSigner. If NULL, defaults
 are applied to all parameters; note however that some parameters do not have
 useful defaults, and will need to be set before signing is attempted.
 @param signingCertificateChain A required CFArrayRef containing the certificate
  chain as SecCertificateRef objects with the leaf certificate at index 0.
 @param flags Optional flags. Pass kSecCSDefaultFlags for standard behavior.
 @param signer On successful return, a SecCodeSignerRemote object reference representing
 the workflow to add a code signature to SecStaticCode objects. On error, unchanged.
 @param error An optional pointer to a CFErrorRef variable. If the call fails
 (something other than errSecSuccess is returned), and this argument is non-NULL,
 a CFErrorRef is stored there further describing the nature and circumstances
 of the failure. The caller must CFRelease() this error object when done with it.
 @result Upon success, errSecSuccess. Upon error, an OSStatus value documented in
 CSCommon.h or certain other Security framework headers.
 */
OSStatus
SecCodeSignerRemoteCreate(CFDictionaryRef parameters,
						  CFArrayRef signingCertificateChain,
						  SecCSFlags flags,
						  SecCodeSignerRemoteRef * CF_RETURNS_RETAINED signer,
						  CFErrorRef *error)
	SPI_AVAILABLE(macos(13.3));


/*!
 @typedef SecCodeRemoteSignHandler
 This is the type of a callback block used to provide the digest hash and digest algorithm to the caller to enable creating a signature.

 @param cmsDigestHash A CFDataRef containing the signature's digest hash that needs to be signed.
 @param digestAlgorithm The digest algorithm used to create the message digest.
 @result Upon success, return a retained CFDataRef representing the signature for the provided message digest that will be released
 upon completion of the signing. Upon error, return NULL to fail the signing operation. Providing a signature that does not validate will
 trigger a validation failure.
 */
typedef CFDataRef (^SecCodeRemoteSignHandler)(CFDataRef cmsDigestHash, SecCSDigestAlgorithm digestAlgorithm);

/*!
 @function SecCodeSignerRemoteAddSignature
 Create a code signature and add it to the StaticCode object being signed, using the remote signing workflow.

 @param signer A SecCodeSignerRemote object containing all the information required
 to produce the code signature.
 @param code A valid SecStaticCode object reference representing code files
 on disk. This code will be signed, and will ordinarily be modified to contain
 the resulting signature data.
 @param flags Optional flags. Pass kSecCSDefaultFlags for standard behavior.
 @param error An optional pointer to a CFErrorRef variable. If the call fails
 (something other than errSecSuccess is returned), and this argument is non-NULL,
 a CFErrorRef is stored there further describing the nature and circumstances
 of the failure. The caller must CFRelease() this error object when done with it.
 @result Upon success, errSecSuccess. Upon error, an OSStatus value documented in
 CSCommon.h or certain other Security framework headers.
 */
OSStatus
SecCodeSignerRemoteAddSignature(SecCodeSignerRemoteRef signer,
								SecStaticCodeRef code,
								SecCSFlags flags,
								SecCodeRemoteSignHandler signHandler,
								CFErrorRef *error)
	SPI_AVAILABLE(macos(13.3));

#ifdef __cplusplus
}
#endif

#endif //_H_SECCODESIGNERREMOTE
