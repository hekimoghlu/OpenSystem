/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#ifndef	_SEC_FDE_RECOVERYASYMMETRIC_CRYPTO_H_
#define _SEC_FDE_RECOVERYASYMMETRIC_CRYPTO_H_

#include <Security/cssmtype.h>
#include <Security/SecBase.h>
#include <CoreFoundation/CFData.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*
	See CEncryptedEncoding.h in the DiskImages project
	This structure is only used in libcsfde/lib/CSRecovery.c and 
	SecFDERecoveryAsymmetricCrypto.cpp
*/

typedef struct 
{
	uint32_t				publicKeyHashSize;
	uint8_t					publicKeyHash[32];
	
	CSSM_ALGORITHMS			blobEncryptionAlgorithm;
	CSSM_PADDING			blobEncryptionPadding;
	CSSM_ENCRYPT_MODE		blobEncryptionMode;
	
	uint32_t				encryptedBlobSize;
	uint8_t					encryptedBlob[512];
} FVPrivateKeyHeader;

int SecFDERecoveryWrapCRSKWithPubKey(const uint8_t *crsk, size_t crskLen, 
	SecCertificateRef certificateRef, FVPrivateKeyHeader *outHeader);
CFDataRef CF_RETURNS_RETAINED SecFDERecoveryUnwrapCRSKWithPrivKey(SecKeychainRef keychain, 
	const FVPrivateKeyHeader *inHeader);

#ifdef  __cplusplus
}
#endif
	
#endif	/* _SEC_FDE_RECOVERYASYMMETRIC_CRYPTO_H_ */
