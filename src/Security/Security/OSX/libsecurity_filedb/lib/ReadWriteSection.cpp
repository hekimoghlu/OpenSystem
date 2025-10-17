/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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

#include "ReadWriteSection.h"

uint32 WriteSection::put(uint32 inOffset, uint32 inValue)
{
	uint32 aLength = CheckUInt32Add(inOffset, sizeof(inValue));
	if (aLength > mCapacity)
		grow(aLength);

	if (mAddress == NULL)
		CssmError::throwMe(CSSMERR_DL_DATABASE_CORRUPT);

	*reinterpret_cast<uint32 *>(mAddress + inOffset) = htonl(inValue);
	return aLength;
}



uint32 WriteSection::put(uint32 inOffset, uint32 inLength, const uint8 *inData)
{
	// if we are being asked to put 0 bytes, just return
	if (inLength == 0 || inData == NULL)
	{
		return inOffset;
	}
	
	uint32 aLength = CheckUInt32Add(inOffset, inLength);
	
	// Round up to nearest multiple of 4 bytes, to pad with zeros
	uint32 aNewOffset = align(aLength);
	if (aNewOffset > mCapacity)
		grow(aNewOffset);

	if (mAddress == NULL)
		CssmError::throwMe(CSSMERR_DL_DATABASE_CORRUPT);

	memcpy(mAddress + inOffset, inData, inLength);

	for (uint32 anOffset = aLength; anOffset < aNewOffset; anOffset++)
		mAddress[anOffset] = 0;

	return aNewOffset;
}



void WriteSection::grow(size_t inNewCapacity)
{
	size_t n = CheckUInt32Multiply((uint32)mCapacity, 2);
	size_t aNewCapacity = max(n, inNewCapacity);
	mAddress = reinterpret_cast<uint8 *>(mAllocator.realloc(mAddress, aNewCapacity));

    if (mAddress == NULL) {
        CssmError::throwMe(CSSMERR_DL_DATABASE_CORRUPT);
    }

    memset(mAddress + mCapacity, 0, aNewCapacity - mCapacity);
    mCapacity = aNewCapacity;
}
