/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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
#ifdef	ASC_CSP_ENABLE

#ifndef _ASC_CONTEXT_H_
#define _ASC_CONTEXT_H_

#include "AppleCSPContext.h"
#include <security_comcryption/comcryption.h>

/* symmetric encrypt/decrypt context */
class ASCContext : public AppleCSPContext {
public:
	ASCContext(AppleCSPSession &session) :
		AppleCSPContext(session),
		mCcObj(NULL)	{ }
	~ASCContext();
	
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
	comcryptObj			mCcObj;
	
	/*
	 * For first implementation, we have to cope with the fact that the final
	 * decrypt call down to the comcryption engine requires *some* ciphertext.
	 * On decrypt, we'll just save one byte on each update in preparation for
	 * the final call. Hopefull we'll have time to fix deComcryptData() so this
	 * is unneccesary.
	 */
	unsigned char		mDecryptBuf;
	bool				mDecryptBufValid;
	
};	/* RC4Context */

#endif 	/*_ASC_CONTEXT_H_ */
#endif	/* ASC_CSP_ENABLE */
