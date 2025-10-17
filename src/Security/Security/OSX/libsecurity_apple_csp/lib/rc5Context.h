/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
 * rc5Context.h - glue between BlockCrytpor and ssleay RC5 implementation
 */
#ifndef _RC5_CONTEXT_H_
#define _RC5_CONTEXT_H_

#include <BlockCryptor.h>
#include <openssl/rc5_legacy.h>

class RC5Context : public BlockCryptor {
public:
	RC5Context(AppleCSPSession &session) :
		BlockCryptor(session)	{ }
	~RC5Context();
	
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
	RC5_32_KEY			rc5Key;
	
};	/* RC5Context */

#endif //_RC2_CONTEXT_H_
