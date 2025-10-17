/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
 * NullCryptor.h - null symmetric encryptor for measurement only
 */
#ifndef _NULL_CRYPTOR_H_
#define _NULL_CRYPTOR_H_

/*
 * DO NOT DEFINE THIS SYMBOL TRUE FOR CODE CHECKED IN TO CVS
 */
#define NULL_CRYPT_ENABLE			0

#if		NULL_CRYPT_ENABLE

#include <security_cdsa_plugin/CSPsession.h>
#include "AppleCSP.h"
#include "AppleCSPContext.h"
#include "AppleCSPSession.h"
#include "BlockCryptor.h"

#define NULL_CRYPT_BLOCK_SIZE		16

class NullCryptor : public BlockCryptor {
public:
	NullCryptor(AppleCSPSession &session) :
		BlockCryptor(session),
		mInitFlag(false)	{ }
	~NullCryptor() { }
	
	// called by CSPFullPluginSession
	void init(const Context &context, bool encoding = true)
	{
		if(mInitFlag && !opStarted()) {
			return;
		}
		/* Just have BlockCryptor do its setup */
		setup(NULL_CRYPT_BLOCK_SIZE, context);
		mInitFlag = true;
	}

	// called by BlockCryptor
	void encryptBlock(
		const void		*plainText,			// length implied (one block)
		size_t			plainTextLen,
		void			*cipherText,	
		size_t			&cipherTextLen,		// in/out, throws on overflow
		bool			final)
	{
		memmove(cipherText, plainText, NULL_CRYPT_BLOCK_SIZE);
		cipherTextLen = NULL_CRYPT_BLOCK_SIZE;
	}
	
	void decryptBlock(
		const void		*cipherText,		// length implied (one cipher block)
		size_t			cipherTextLen,
		void			*plainText,	
		size_t			&plainTextLen,		// in/out, throws on overflow
		bool			final)
	{
		memmove(plainText, cipherText, NULL_CRYPT_BLOCK_SIZE);
		plainTextLen = NULL_CRYPT_BLOCK_SIZE;
	}
		
private:
	bool				mInitFlag;			// for easy reuse

};	/* NullCryptor */

#endif	/* NULL_CRYPT_ENABLE */

#endif //_NULL_CRYPTOR_H_
