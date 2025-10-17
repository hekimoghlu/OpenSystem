/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 8, 2025.
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
 * rc4Context.h - glue between BlockCrytpor and ssleay RC4 implementation
 */
#ifndef _RC4_CONTEXT_H_
#define _RC4_CONTEXT_H_

#include "AppleCSPContext.h"
#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonCryptorSPI.h>

class RC4Context : public AppleCSPContext {
public:
	RC4Context(AppleCSPSession &session) :
		AppleCSPContext(session)	{ }
	virtual ~RC4Context();
	
	// called by CSPFullPluginSession
	void init(
		const Context 	&context, 
		bool encoding = true);
	void update(
		void 			*inp, 
		size_t 			&inSize, 			// in/out
		void 			*outp, 
		size_t 			&outSize);			// in/out
	void final(
		CssmData 		&out);

 	size_t inputSize(
		size_t 			outSize);			// input for given output size
	size_t outputSize(
		bool 			final = false, 
		size_t 			inSize = 0); 		// output for given input size
	void minimumProgress(
		size_t 			&in, 
		size_t 			&out); 				// minimum progress chunks
	
private:
    CCCryptorRef    rc4Key;
	
};	/* RC4Context */

#endif //_RC4_CONTEXT_H_
