/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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
 * desContext.h - glue between BlockCrytpor and DES/3DES implementations
 */
#ifndef _DES_CONTEXT_H_
#define _DES_CONTEXT_H_

#include <security_cdsa_plugin/CSPsession.h>
#include "AppleCSP.h"
#include "AppleCSPContext.h"
#include "AppleCSPSession.h"
#include "BlockCryptor.h"
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonCryptorSPI.h>

#define DES_KEY_SIZE_BITS_EXTERNAL		(kCCKeySizeDES * 8)
#define DES_BLOCK_SIZE_BYTES			kCCBlockSizeDES

/* DES Symmetric encryption context */
class DESContext : public BlockCryptor {
public:
	DESContext(AppleCSPSession &session);
	virtual ~DESContext();
	
	// called by CSPFullPluginSession
	void init(const Context &context, bool encoding = true);

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
        CCCryptorRef	DesInst;	
};	/* DESContext */

/* Triple-DES (EDE, 24 byte key) Symmetric encryption context */

#define DES3_KEY_SIZE_BYTES		(3 * (DES_KEY_SIZE_BITS_EXTERNAL / 8))
#define DES3_BLOCK_SIZE_BYTES	kCCBlockSize3DES

class DES3Context : public BlockCryptor {
public:
	DES3Context(AppleCSPSession &session);
	~DES3Context();
	
	// called by CSPFullPluginSession
	void init(const Context &context, bool encoding = true);

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
    CCCryptorRef	DesInst;		
};	/* DES3Context */

#endif //_DES_CONTEXT_H_
