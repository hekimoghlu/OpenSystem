/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
 */#include "HIDLib.h"

//#include <stdio.h>

/*
 *------------------------------------------------------------------------------
 *
 * HIDPutData - Put a single data item to a report
 *
 *	 Input:
 *			  psReport				- The report
 *			  iReportLength			- The length of the report
 *			  iStart				- Start Bit in report
 *			  iSize					- Number of Bits
 *			  iValue				- The data
 *	 Output:
 *	 Returns:
 *			  kHidP_Success			- Success
 *			  kHidP_NullPointer		- Argument, Pointer was Null
 *
 *------------------------------------------------------------------------------
*/
OSStatus
HIDPutData				   (void *					report,
							IOByteCount				reportLength,
							UInt32					start,
							UInt32					size,
							SInt32 					value)
{
	UInt8 * psReport = (UInt8 *)report;
	SInt32 data, iShiftedData;
	UInt32 iStartByte, startBit;
	UInt32 iLastByte, iLastBit;
	UInt32 iStartMask, iLastMask;
	UInt32 iDataMask;
/*
 *	  Report
 *	  Bit 28 27 26 25 24 | 23 22 21 20 19 18 17 16 | 15 14 13 12 11 10 09 ...
 *	  Last Byte (3) |	 |		  Byte 2		   |	 |	Start Byte (1)
 *	  Data x  x	 x	d  d |	d  d  d	 d	d  d  d	 d |  d	 d	y  y  y	 y	y
 *	  Last Bit (1) /	 |						   |	  \ Start Bit (6)
 *	  ...  1  1	 1	0  0 |	   Intermediate		   |  0	 0	1  1  1	 1	1 ...
 *	  Last Mask			 |		 Byte(s)		   |		StartMask
*/
	iLastByte = (start + size - 1)/8;
/*
 *	Check the parameters
*/
	if ((size == 0) || (iLastByte >= reportLength))
		return kHIDBadParameterErr;
	iLastBit = (start + size - 1)&7;
	iLastMask = ~((1<<(iLastBit+1)) - 1);
	iStartByte = start/8;
	startBit = start&7;
	iStartMask = (1<<startBit) - 1;
/*
 *	If the data is contained in one byte then
 *	  handle it differently
 *	  Mask off just the area where the new data goes
 *	  Shift the data over to its new location
 *	  Mask the data for its new location
 *	  Or in the data
*/
	if (iStartByte == iLastByte)
	{
		data = psReport[iStartByte];
		iDataMask = iStartMask | iLastMask;
		data &= iDataMask;
		iShiftedData = value << startBit;
		iShiftedData &= ~iDataMask;
		data |= iShiftedData;
	}
/*
 *	If the data is in more than one byte then
 *	Do the start byte first
 *	Mask off the bits where the new data goes
 *	Shift the new data over to the start of field
 *	Or the two together and store back out
*/
	else
	{
		data = psReport[iStartByte];
		data &= iStartMask;
		iShiftedData = value << startBit;
		data |= iShiftedData;
		psReport[iStartByte] = (UInt8) data;
		iShiftedData >>= 8;
/*
 *		Store out an intermediate bytes
*/
		while (++iStartByte < iLastByte)
		{
			psReport[iStartByte] = (UInt8) iShiftedData;
			iShiftedData >>= 8;
		}
/*
 *		Mask off the bits where the new data goes
 *		Mask off the bits in the new data where the old goes
 *		Or the two together and store back out
*/
		data = psReport[iLastByte];
		data &= iLastMask;
		iShiftedData &= ~iLastMask;
		data |= iShiftedData;
	}
/*
 *	Store out the last or only Byte
*/
	psReport[iStartByte] = (UInt8) data;
	return kHIDSuccess;
}


