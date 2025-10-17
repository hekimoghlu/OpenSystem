/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
#ifndef	_CK_FEEECDSA_H_
#define _CK_FEEECDSA_H_

#if	!defined(__MACH__)
#include <ckconfig.h>
#include <feeTypes.h>
#include <feePublicKey.h>
#else
#include <security_cryptkit/ckconfig.h>
#include <security_cryptkit/feeTypes.h>
#include <security_cryptkit/feePublicKey.h>
#endif

/* 
 * Keep this one defined and visible even if we can't actually do ECDSA - feeSigParse()
 * uses it to detect "wriong signature type".
 */
#define FEE_ECDSA_MAGIC		0xfee00517

#ifdef __cplusplus
extern "C" {
#endif


/*
 * Sign specified block of data (most likely a hash result) using
 * specified private key. Result, an enc64-encoded signature block,
 * is returned in *sigData.
 */
feeReturn feeECDSASign(feePubKey pubKey,
    feeSigFormat  format,         // Format of the signature DER/RAW
	const unsigned char *data,   	// data to be signed
	unsigned dataLen,				// in bytes
	feeRandFcn randFcn,				// optional
	void *randRef,					// optional 
	unsigned char **sigData,		// malloc'd and RETURNED
	unsigned *sigDataLen);			// RETURNED

/*
 * Verify signature, obtained via feeECDSASign, for specified
 * data (most likely a hash result) and feePubKey. Returns FR_Success or
 * FR_InvalidSignature.
 */
feeReturn feeECDSAVerify(const unsigned char *sigData,
	size_t sigDataLen,
	const unsigned char *data,
	unsigned dataLen,
	feePubKey pubKey,
    feeSigFormat  format);        // Format of the signature DER/RAW

/*
 * For given key, calculate maximum signature size. 
 */
feeReturn feeECDSASigSize(
	feePubKey pubKey,
	unsigned *maxSigLen);

#ifdef __cplusplus
}
#endif

#endif	/*_CK_FEEECDSA_H_*/
