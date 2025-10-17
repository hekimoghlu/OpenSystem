/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
 * DigestObject.cpp - generic C++ implementations of SHA1 and MD5. 
 *
 */

#include "SHA1_MD5_Object.h"
#include <stdexcept>
#include <string.h>

/***
 *** MD5
 ***/
void MD5Object::digestInit()
{
	mIsDone = false;
	CC_MD5_Init(&mCtx);
}

void MD5Object::digestUpdate(
	const void 	*data, 
	size_t 		len)
{
	if(mIsDone) {
		throw std::runtime_error("MD5 digestUpdate after final");
	}
	CC_MD5_Update(&mCtx, data, (CC_LONG)len);
}

void MD5Object::digestFinal(
	void 		*digest)
{
	if(mIsDone) {
		throw std::runtime_error("MD5 digestFinal after final");
	}
	CC_MD5_Final((unsigned char *)digest, &mCtx);
	mIsDone = true;
}

/* use default memberwise init */
DigestObject *MD5Object::digestClone() const
{
	return new MD5Object(*this);
}

size_t MD5Object::digestSizeInBytes() const
{
	return CC_MD5_DIGEST_LENGTH;
}

/***
 *** SHA1
 ***/
void SHA1Object::digestInit()
{
	mIsDone = false;
	CC_SHA1_Init(&mCtx);
}

void SHA1Object::digestUpdate(
	const void 	*data, 
	size_t 		len)
{
	CC_SHA1_Update(&mCtx, (const unsigned char *)data, (CC_LONG)len);
}

void SHA1Object::digestFinal(
	void 		*digest)
{
	CC_SHA1_Final((unsigned char *)digest, &mCtx);
	mIsDone = true;
}

/* use default memberwise init */
DigestObject *SHA1Object::digestClone() const
{
	return new SHA1Object(*this);
}

size_t SHA1Object::digestSizeInBytes() const
{
	return CC_SHA1_DIGEST_LENGTH;
}

