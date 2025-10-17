/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#ifndef __PTRAUTH_UTILS_H
#define __PTRAUTH_UTILS_H

#include <ptrauth.h>
#include <sys/cdefs.h>
__BEGIN_DECLS

/* ptrauth_utils flags */
#define PTRAUTH_ADDR_DIVERSIFY  0x0001  /* Mix storage address in to signature */
#define PTRAUTH_NON_NULL        0x0002  /* ptr must not be NULL */

/* ptrauth_utils_sign_blob_generic
 *
 * Description:	Sign a blob of data with the GA key and extra data, optionally
 * diversified by its storage address.
 *
 * WARNING: Lower 32 bits are always zeroes.
 *
 * Caveat: A race window exists between the blob being written to memory and its signature being
 * calculated by this function. In normal operation, standard thread safety semantics prevent this being
 * an issue, however in the malicious case it should be acknowledged that an attacker may be able to accurately
 * time overwriting parts/all of the blob and we would generate a signature for that modified data. It is
 * therefore important that users of this API minimise that window by calculating signatures immediately
 * after modification to the blob.
 *
 *
 * Parameters:	ptr				Address of data to sign
 *				len_bytes		Length in bytes of data to sign
 *				data			Salt to mix in signature when signing
 *				flags               Signing options
 *
 * Returns:		ptrauth_generic_signature_t		Signature of blob
 *
 */
ptrauth_generic_signature_t
ptrauth_utils_sign_blob_generic(const void * ptr, size_t len_bytes, uint64_t data, int flags);


/* ptrauth_utils_auth_blob_generic
 *
 * Description:	Authenticates a signature for a blob of data
 *
 * Caveat: As with ptrauth_utils_sign_blob_generic, an attacker who is able to accurately time access between
 * authenticating blobs and its use may be able to modify its contents. Failure to time this correctly will
 * result in a panic. Care should be taken to authenticate immediately before reading data from the blob to
 * minimise this window.
 *
 * Parameters:	ptr				Address of data being authenticated
 *				len_bytes		Length of data being authenticated
 *				data			Salt to mix with digest when authenticating
 *				flags           Signing options
 *				signature		The signature to verify
 *
 * Returns:		void			If the function returns, the authentication succeeded,
 *								else we panic as something's gone awry
 *
 */
void
ptrauth_utils_auth_blob_generic(const void * ptr, size_t len_bytes, uint64_t data, int flags, ptrauth_generic_signature_t signature);

__END_DECLS
#endif // __PTRAUTH_UTILS_H
