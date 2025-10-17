/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#ifndef _CC_DH_H_
#define _CC_DH_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_MSC_VER)
#include <availability.h>
#else
#include <os/availability.h>
#endif


#ifdef __cplusplus
extern "C" {
#endif

typedef struct CCDHRef_s *CCDHRef;

/*!
    @enum       CCDHParameters
    @abstract   Diffie-Hellman parameters.

    @constant   kCCDHRFC3526Group5
*/
enum {
    // 1536 bit DL group
    kCCDHRFC3526Group5 = 1
};

typedef uint32_t CCDHParameters
API_AVAILABLE(macos(10.8), ios(5.0));

/*!
    @function   CCDHCreate
    @abstract   Creates a Diffie-Hellman context.

	@param      dhParameter  The Diffie-Hellman Group to use (provides p and g).
                             The only accepted value is kCCDHRFC3526Group5.

    @result     If unable to allocate memory this returns NULL.
*/
CCDHRef
CCDHCreate(CCDHParameters dhParameter)
API_AVAILABLE(macos(10.8), ios(5.0));

/*!
    @function   CCDHRelease
    @abstract   Releases a Diffie-Hellman context.

	@param      ref  The Diffie-Hellman context to clear and deallocate.

*/
void
CCDHRelease(CCDHRef ref)
API_AVAILABLE(macos(10.8), ios(5.0));

/*!
    @function   CCDHGenerateKey
    @abstract   Generate the public key for use in a Diffie-Hellman handshake.

	@param      ref  The Diffie-Hellman context.
    @result		returns -1 on failure.

*/
int
CCDHGenerateKey(CCDHRef ref, void *output, size_t *outputLength)
API_AVAILABLE(macos(10.8), ios(5.0));

/*!
    @function   CCDHComputeKey
    @abstract   Compute the shared Diffie-Hellman key using the peer's public
                key.

	@param      sharedKey  Shared key computed from the peer public key, p, g,
                            and the private key.
	@param      peerPubKey  Public key received from the peer.
    @param		peerPubKeyLen Length of peer public key.
	@param      ref  The Diffie-Hellman context.

	@returns    length of the shared key.

*/
int
CCDHComputeKey(unsigned char *sharedKey, size_t *sharedKeyLen, const void *peerPubKey, size_t peerPubKeyLen, CCDHRef ref)
API_AVAILABLE(macos(10.8), ios(5.0));

#ifdef __cplusplus
}
#endif

#endif  /* _CC_DH_H_ */
