/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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
// gladmanContext.h - Gladman AES context class
//
#ifndef _H_GLADMAN_CONTEXT
#define _H_GLADMAN_CONTEXT

#include <security_cdsa_plugin/CSPsession.h>
#include "AppleCSP.h"
#include "AppleCSPContext.h"
#include "AppleCSPSession.h"
#include "BlockCryptor.h"
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonCryptorSPI.h>
#include "aesCommon.h"

#define GLADMAN_BLOCK_SIZE_BYTES	DEFAULT_AES_BLOCK_BYTES

/* Symmetric encryption context */
class GAESContext : public BlockCryptor {
public:
	GAESContext(AppleCSPSession &session);
	virtual ~GAESContext();
	
	// called by CSPFullPluginSession
	void init(const Context &context, bool encoding = true);

	// As an optimization, we allow reuse of a modified context. The main thing
	// we avoid is a redundant key scheduling. We save the current raw keys bits
	// in mRawKey and compare on re-init.
	bool changed(const Context &context)	 { return true; }

	// called by BlockCryptor
	void encryptBlock(
		const void		*plainText,			// length implied (one block)
		size_t			plainTextLen,
		void			*cipherText,	
		size_t			&cipherTextLen,		// in/out, throws on overflow
		bool			final);
	void decryptBlock(
		const void		*cipherText,		// length implied (one cipher block)
		size_t			cipherTextLen,	
		void			*plainText,	
		size_t			&plainTextLen,		// in/out, throws on overflow
		bool			final);
	
private:
	void deleteKey();
	
	/* scheduled key */
    CCCryptorRef	mAesKey;	
	bool				mInitFlag;			// for easy reuse
	
	/* 
	 * Raw key bits saved here and checked on re-init to avoid extra key 
	 * schedule on re-init. We also have to do a new key schedule if 
	 * changing between encrypting and decrypting since the key schedules
	 * differ for the two.
	 */
	uint8				mRawKey[MAX_AES_KEY_BITS / 8];
	uint32				mRawKeySize;
	bool				mWasEncrypting;
};	/* AESContext */

#endif //_H_GLADMAN_CONTEXT
