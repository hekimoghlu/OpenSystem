/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 12, 2024.
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
/*
 * castContext.h - glue between BlockCrytpor and ssleay CAST-128 (CAST5)
 *				 implementation
 *
 *
 * Here's what RFC 2144 has to say about CAST128 and CAST5 nomenclature:
 *
 *    In order to avoid confusion when variable keysize operation is 
 *    used, the name CAST-128 is to be considered synonymous with the 
 *    name CAST5; this allows a keysize to be appended without ambiguity.  
 *    Thus, for example, CAST-128 with a 40-bit key is to be referred to 
 *    as CAST5-40; where a 128-bit key is explicitly intended, the 
 *    name CAST5-128 should be used. 
 *
 * This module implements a variable key length, from 40 bits to 128 bits,
 * and can thus be said to implement both CAST-128 and CAST5.
 */
 
#ifndef _CAST_CONTEXT_H_
#define _CAST_CONTEXT_H_

#include "AppleCSPContext.h"
#include "BlockCryptor.h"
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonCryptorSPI.h>

class CastContext : public BlockCryptor {
public:
	CastContext(AppleCSPSession &session);
	virtual ~CastContext();
	
	// called by CSPFullPluginSession
	void init(const Context &context, bool encoding = true);

	// As an optimization, we allow reuse of a modified context. 
	// The main thing we avoid is a redundant key scheduling. We 
	// save the current raw keys bits in mRawKey and compare on 
	// re-init.
	bool changed(const Context &context)	 { return true; }

	// called by BlockCryptor
	void encryptBlock(
		const void		*plainText,		// length implied (one block)
		size_t			plainTextLen,
		void			*cipherText,	
		size_t			&cipherTextLen,	// in/out, throws on overflow
		bool			final);
	void decryptBlock(
		const void		*cipherText,	// length implied (one cipher block)
		size_t			cipherTextLen,
		void			*plainText,	
		size_t			&plainTextLen,	// in/out, throws on overflow
		bool			final);
	
private:
	void deleteKey();

	/* scheduled key */
    CCCryptorRef	mCastKey;		

	bool				mInitFlag;			// for easy reuse

	/* 
	 * Raw key bits saved here and checked on re-init to avoid 
	 * extra key schedule 
	 */
	uint8				mRawKey[kCCKeySizeMaxCAST];
	uint32				mRawKeySize;

	
};	/* CastContext */

#endif //_CAST_CONTEXT_H_
