/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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
 * SHA2_Object.cpp - SHA2 digest objects 
 */

#include "SHA2_Object.h"
#include <stdexcept>
#include <string.h>

/***
 *** SHA224
 ***/
void SHA224Object::digestInit()
{
	mIsDone = false;
	CC_SHA224_Init(&mCtx);
}

void SHA224Object::digestUpdate(
	const void 	*data, 
	size_t 		len)
{
	CC_SHA224_Update(&mCtx, (const unsigned char *)data, (CC_LONG)len);
}

void SHA224Object::digestFinal(
	void 		*digest)
{
	CC_SHA224_Final((unsigned char *)digest, &mCtx);
	mIsDone = true;
}

/* use default memberwise init */
DigestObject *SHA224Object::digestClone() const
{
	return new SHA224Object(*this);
}

size_t SHA224Object::digestSizeInBytes() const
{
	return CC_SHA224_DIGEST_LENGTH;
}

/***
 *** SHA256
 ***/
void SHA256Object::digestInit()
{
	mIsDone = false;
	CC_SHA256_Init(&mCtx);
}

void SHA256Object::digestUpdate(
	const void 	*data, 
	size_t 		len)
{
	CC_SHA256_Update(&mCtx, (const unsigned char *)data, (CC_LONG)len);
}

void SHA256Object::digestFinal(
	void 		*digest)
{
	CC_SHA256_Final((unsigned char *)digest, &mCtx);
	mIsDone = true;
}

/* use default memberwise init */
DigestObject *SHA256Object::digestClone() const
{
	return new SHA256Object(*this);
}

size_t SHA256Object::digestSizeInBytes() const
{
	return CC_SHA256_DIGEST_LENGTH;
}

/***
 *** SHA384
 ***/
void SHA384Object::digestInit()
{
	mIsDone = false;
	CC_SHA384_Init(&mCtx);
}

void SHA384Object::digestUpdate(
	const void 	*data, 
	size_t 		len)
{
	CC_SHA384_Update(&mCtx, (const unsigned char *)data, (CC_LONG)len);
}

void SHA384Object::digestFinal(
	void 		*digest)
{
	CC_SHA384_Final((unsigned char *)digest, &mCtx);
	mIsDone = true;
}

/* use default memberwise init */
DigestObject *SHA384Object::digestClone() const
{
	return new SHA384Object(*this);
}

size_t SHA384Object::digestSizeInBytes() const
{
	return CC_SHA384_DIGEST_LENGTH;
}

/***
 *** SHA512
 ***/
void SHA512Object::digestInit()
{
	mIsDone = false;
	CC_SHA512_Init(&mCtx);
}

void SHA512Object::digestUpdate(
	const void 	*data, 
	size_t 		len)
{
	CC_SHA512_Update(&mCtx, (const unsigned char *)data, (CC_LONG)len);
}

void SHA512Object::digestFinal(
	void 		*digest)
{
	CC_SHA512_Final((unsigned char *)digest, &mCtx);
	mIsDone = true;
}

/* use default memberwise init */
DigestObject *SHA512Object::digestClone() const
{
	return new SHA512Object(*this);
}

size_t SHA512Object::digestSizeInBytes() const
{
	return CC_SHA512_DIGEST_LENGTH;
}

