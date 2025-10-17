/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 20, 2022.
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
 * MD2Object.cpp
 */
#include "MD2Object.h"
#include <stdexcept>
#include <string.h>

void MD2Object::digestInit()
{
	setIsDone(false);
	CC_MD2_Init(&mCtx);
}

void MD2Object::digestUpdate(
	const void 	*data, 
	size_t 		len)
{
	if(isDone()) {
		throw std::runtime_error("MD2 digestUpdate after final");
	}
	CC_MD2_Update(&mCtx, (unsigned char *)data, (CC_LONG)len);
}

void MD2Object::digestFinal(
	void 		*digest)
{
	if(isDone()) {
		throw std::runtime_error("MD2 digestFinal after final");
	}
	CC_MD2_Final((unsigned char *)digest, &mCtx);
	setIsDone(true);
}

/* use default memberwise init */
DigestObject *MD2Object::digestClone() const
{
	return new MD2Object(*this);
}

CSSM_SIZE MD2Object::digestSizeInBytes() const
{
	return CC_MD2_DIGEST_LENGTH;
}

