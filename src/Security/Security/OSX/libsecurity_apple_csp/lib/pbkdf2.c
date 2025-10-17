/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 25, 2025.
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
 	File:		pbkdf2.c
 	Contains:	Apple Data Security Services PKCS #5 PBKDF2 function definition.
 	Copyright (c) 1999,2011,2013-2014 Apple Inc. All Rights Reserved.
*/
#include "pbkdf2.h"
#include <ConditionalMacros.h>
#include <string.h>
/* Will write hLen bytes into dataPtr according to PKCS #5 2.0 spec.
   See: http://www.rsa.com/rsalabs/pubs/PKCS/html/pkcs-5.html for details. 
   tempBuffer is a pointer to at least MAX (hLen, saltLen + 4) + hLen bytes. */
static void 
F (PRF prf, uint32 hLen,
   const void *passwordPtr, uint32 passwordLen,
   const void *saltPtr, uint32 saltLen,
   uint32 iterationCount,
   uint32 blockNumber,
   void *dataPtr,
   void *tempBuffer)
{
	uint8 *inBlock, *outBlock, *resultBlockPtr;
	uint32 iteration;
	outBlock = (uint8*)tempBuffer;
	inBlock = outBlock + hLen;
	/* Set up inBlock to contain Salt || INT (blockNumber). */
	memcpy (inBlock, saltPtr, saltLen);

	inBlock[saltLen + 0] = (uint8)(blockNumber >> 24);
	inBlock[saltLen + 1] = (uint8)(blockNumber >> 16);
	inBlock[saltLen + 2] = (uint8)(blockNumber >> 8);
	inBlock[saltLen + 3] = (uint8)(blockNumber);

	/* Caculate U1 (result goes to outBlock) and copy it to resultBlockPtr. */
	resultBlockPtr = (uint8*)dataPtr;
	prf (passwordPtr, passwordLen, inBlock, saltLen + 4, outBlock);
	memcpy (resultBlockPtr, outBlock, hLen);
	/* Calculate U2 though UiterationCount. */
	for (iteration = 2; iteration <= iterationCount; iteration++)
	{
		uint8 *tempBlock;
		uint32 byte;
		/* Swap inBlock and outBlock pointers. */
		tempBlock = inBlock;
		inBlock = outBlock;
		outBlock = tempBlock;
		/* Now inBlock conatins Uiteration-1.  Calclulate Uiteration into outBlock. */
		prf (passwordPtr, passwordLen, inBlock, hLen, outBlock);
		/* Xor data in dataPtr (U1 \xor U2 \xor ... \xor Uiteration-1) with
		   outBlock (Uiteration). */
		for (byte = 0; byte < hLen; byte++)
			resultBlockPtr[byte] ^= outBlock[byte];
	}
}
void pbkdf2 (PRF prf, uint32 hLen,
			 const void *passwordPtr, uint32 passwordLen,
			 const void *saltPtr, uint32 saltLen,
			 uint32 iterationCount,
			 void *dkPtr, uint32 dkLen,
			 void *tempBuffer)
{
	uint32 completeBlocks = dkLen / hLen;
	uint32 partialBlockSize = dkLen % hLen;
	uint32 blockNumber;
	uint8 *dataPtr = (uint8*)dkPtr;
	uint8 *blkBuffer = (uint8*)tempBuffer;
	/* First calculate all the complete hLen sized blocks required. */
	for (blockNumber = 1; blockNumber <= completeBlocks; blockNumber++)
	{
		F (prf, hLen, passwordPtr, passwordLen, saltPtr, saltLen,
		   iterationCount, blockNumber, dataPtr, blkBuffer + hLen);
		dataPtr += hLen;
	}
	/* Finally if the requested output size was not an even multiple of hLen, calculate
	   the final block and copy the first partialBlockSize bytes of it to the output. */
	if (partialBlockSize > 0)
	{
		F (prf, hLen, passwordPtr, passwordLen, saltPtr, saltLen,
		   iterationCount, blockNumber, blkBuffer, blkBuffer + hLen);
		memcpy (dataPtr, blkBuffer, partialBlockSize);
	}
}
