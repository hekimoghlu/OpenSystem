/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#ifndef	_CK_FEEDIGITALSIG_H_
#define _CK_FEEDIGITALSIG_H_

#if	!defined(__MACH__)
#include <feeTypes.h>
#include <feePublicKey.h>
#else
#include <security_cryptkit/feeTypes.h>
#include <security_cryptkit/feePublicKey.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define FEE_SIG_MAGIC		0xfee00516

/*
 * Opaque signature handle.
 */
typedef void *feeSig;

/*
 * Create new feeSig object, including a random large integer 'Pm' for
 * possible use in salting a feeHash object.
 */
feeSig feeSigNewWithKey(
	feePubKey 		pubKey,
	feeRandFcn		randFcn,		/* optional */
	void			*randRef);		/* optional */

void feeSigFree(
	feeSig 			sig);

/*
 * Obtain a malloc'd Pm after or feeSigNewWithKey() feeSigParse()
 */
unsigned char *feeSigPm(
	feeSig 			sig,
	unsigned 		*PmLen);		/* RETURNED */

/*
 * Sign specified block of data (most likely a hash result) using
 * specified feePubKey.
 */
feeReturn feeSigSign(
	feeSig 			sig,
	const unsigned char	*data,   	// data to be signed
	unsigned 		dataLen,	// in bytes
	feePubKey 		pubKey);

/*
 * Given a feeSig processed by feeSigSign, obtain a malloc'd byte
 * array representing the signature.
 */
feeReturn feeSigData(
	feeSig 			sig,
	unsigned char 		**sigData,	// malloc'd and RETURNED
	unsigned 		*sigDataLen);	// RETURNED

/*
 * Obtain a feeSig object by parsing an existing signature block.
 * Note that if Pm is used to salt a hash of the signed data, this must
 * be performed prior to hashing.
 */
feeReturn feeSigParse(
	const unsigned char	*sigData,
	size_t			sigDataLen,
	feeSig 			*sig);		// RETURNED

/*
 * Verify signature, obtained via feeSigParse, for specified
 * data (most likely a hash result) and feePubKey. Returns FR_Success or
 * FR_InvalidSignature.
 */
feeReturn feeSigVerify(
	feeSig 			sig,
	const unsigned char	*data,
	unsigned 		dataLen,
	feePubKey 		pubKey);

/*
 * For given key, calculate maximum signature size. 
 */
feeReturn feeSigSize(
	feePubKey		pubKey,
	unsigned 		*maxSigLen);

#ifdef __cplusplus
}
#endif

#endif	/*_CK_FEEDIGITALSIG_H_*/
