/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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

#ifndef __OVERUNDERFLOWCHECK__
#define __OVERUNDERFLOWCHECK__

inline uint32 CheckUInt32Add(uint32 a, uint32 b)
{
	uint32 c = a + b;
	if (c < a)	
	{
		CssmError::throwMe(CSSMERR_DL_DATABASE_CORRUPT);
	}
	
	return c;
}



inline uint32 CheckUInt32Subtract(uint32 a, uint32 b)
{
	if (a < b)
	{
		CssmError::throwMe(CSSMERR_DL_DATABASE_CORRUPT);
	}

	return a - b;
}



inline uint32 CheckUInt32Multiply(uint32 a, uint32 b)
{
	uint32 c = a * b;
	uint64 cc = ((uint64) a) * ((uint64) b);
	if (c != cc)
	{
		CssmError::throwMe(CSSMERR_DL_DATABASE_CORRUPT);
	}
	
	return c;
}



inline uint64 Check64BitAdd(uint64 a, uint64 b)
{
	uint64 c = a + b;
	if (c < a)
	{
		CssmError::throwMe(CSSMERR_DL_DATABASE_CORRUPT);
	}
	
	return c;
}



inline uint64 Check64BitSubtract(uint64 a, uint64 b)
{
	if (a < b)
	{
		CssmError::throwMe(CSSMERR_DL_DATABASE_CORRUPT);
	}

	return a - b;
}


	
inline uint64 Check64BitMultiply(uint64 a, uint64 b)
{
	if (a != 0)
	{
		uint64 max = (uint64) -1;
		uint64 limit = max / a;
		if (b > limit)
		{
			CssmError::throwMe(CSSMERR_DL_DATABASE_CORRUPT);
		}
	}
	
	return a * b;
}



#endif
