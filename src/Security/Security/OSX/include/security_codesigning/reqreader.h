/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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
// reqreader - Requirement language (exprOp) reader/scanner
//
#ifndef _H_REQREADER
#define _H_REQREADER

#include "requirement.h"
#include <Security/SecCertificate.h>

namespace Security {
namespace CodeSigning {


//
// The Reader class provides structured access to a opExpr-type code requirement.
//
class Requirement::Reader {
public:
	Reader(const Requirement *req);
	
	const Requirement *requirement() const { return mReq; }
	
	template <class T> T get();
	void getData(const void *&data, size_t &length);
	
	std::string getString();
	CFDataRef getHash();
	CFAbsoluteTime getAbsoluteTime();
	const unsigned char *getSHA1();
	
	template <class T> void getData(T *&data, size_t &length)
	{ return getData(reinterpret_cast<const void *&>(data), length); }

protected:
	void checkSize(size_t length)
	{
		if (mPC + length < mPC || mPC + length > mReq->length())
			MacOSError::throwMe(errSecCSReqInvalid);
	}
	
	void skip(size_t length);
	
	Offset pc() const { return mPC; }
	bool atEnd() const { return mPC >= mReq->length(); }
	
private:
	const Requirement * const mReq;
	Offset mPC;
};

template <class T>
T Requirement::Reader::get()
{
	checkSize(sizeof(T));
	const Endian<const T> *value = mReq->at<Endian<const T> >(mPC);
	mPC += sizeof(T);
	return *value;
}


}	// CodeSigning
}	// Security

#endif //_H_REQREADER
