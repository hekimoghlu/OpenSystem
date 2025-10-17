/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
 * appleUtils.h - Support for ssleay-derived crypto modules
 */
 
#ifndef	_OPENSSL_UTILS_H_
#define _OPENSSL_UTILS_H_

#include <openssl/opensslerr.h>

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * Trivial exception class associated with an openssl error.
 */
class openSslException
{
public:
	openSslException(
		int irtn,
		const char *op = NULL); 	
	~openSslException() 				{ }
	int irtn()	{ return mIrtn; }
private:
	int mIrtn;
};

/* Clear openssl error stack. */
void clearOpensslErrors();

unsigned long logSslErrInfo(const char *op);

void throwRsaDsa(
	const char *op) __attribute__((analyzer_noreturn));
	
/*
 * given an openssl-style error, throw appropriate CssmError.
 */
void throwOpensslErr(
	int irtn);


#ifdef	__cplusplus
}
#endif

#endif	/* _OPENSSL_UTILS_H_ */
