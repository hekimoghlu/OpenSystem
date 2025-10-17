/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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
 * Date: Monday, January 20, 2025.
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

/*
 *-----------------------------------------------------------------------------
 *
 * HIDNextItem - Get the Next Item
 *
 *	 Input:
 *			  ptDescriptor			- Descriptor Structure
 *	 Output:
 *			  ptItem				- Caller-provided Item Structure
 *	 Returns:
 *			  kHIDSuccess		  - Success
 *			  kHIDEndOfDescriptorErr - End of the HID Report Descriptor
 *
 *-----------------------------------------------------------------------------
*/
OSStatus HIDNextItem(HIDReportDescriptor *ptDescriptor)
{
	HIDItem *ptItem;
	unsigned char iHeader;
	unsigned char *psD;
	int i;
	int iLength;
	UInt32 *piX;
	int iSize;
	int iByte = 0;
/*
 *	Disallow Null Pointers
*/
	if (ptDescriptor==NULL)
    {
        return kHIDNullPointerErr;
    }
/*
 *	Use local pointers
*/
	ptItem = &ptDescriptor->item;
	psD = ptDescriptor->descriptor;
	piX = &ptDescriptor->index;
	iLength = (int)ptDescriptor->descriptorLength;
/*
 *	Don't go past the end of the buffer
*/
	if (*piX >= (UInt32)iLength)
    {
        return kHIDEndOfDescriptorErr;
    }
/*
 *	Get the header byte
*/
	iHeader = psD[(*piX)++];
/*
 *	Don't go past the end of the buffer
*/
	if (*piX > (UInt32)iLength)
    {
        return kHIDEndOfDescriptorErr;
    }
	ptItem->itemType = iHeader;
	ptItem->itemType &= kHIDItemTypeMask;
	ptItem->itemType >>= kHIDItemTypeShift;
/*
 *	Long Item Header
 *	Skip Long Items!
 *  If this is a long item header, it's possible we could have an out of bounds access here,
 *  so check if we are at the boundary (rdar://131048343 (Off-by-one when parsing HID Items)).
*/
	if (iHeader == kHIDLongItemHeader)
	{
        if (*piX == (UInt32)iLength)
        {
            return kHIDEndOfDescriptorErr;
        }
		iSize = psD[(*piX)++];
		ptItem->tag = (*piX)++;
	}
/*
 *	Short Item Header
*/
	else
	{
		iSize = iHeader;
		iSize &= kHIDItemSizeMask;
		if (iSize == 3)
        {
            iSize = 4;
        }
		ptItem->byteCount = iSize;
		ptItem->tag = iHeader;
		ptItem->tag &= kHIDItemTagMask;
		ptItem->tag >>= kHIDItemTagShift;
	}
/*
 *	Don't go past the end of the buffer
*/
	if ((*piX + iSize) > (UInt32)iLength)
    {
        return kHIDEndOfDescriptorErr;
    }
/*
 *	Pick up the data
*/
	ptItem->unsignedValue = 0;
	if (iSize == 0)
	{
		ptItem->signedValue = 0;
		return kHIDSuccess;
	}
/*
 *	Get the data bytes
*/
	for (i = 0; i < iSize; i++)
	{
		iByte = psD[(*piX)++];
		ptItem->unsignedValue |= (iByte << (i * 8));
	}
/*
 *	Keep one value unsigned
*/
	ptItem->signedValue = ptItem->unsignedValue;
/*
 *	Sign extend one value
*/
	if ((iByte & 0x80) != 0)
	{
		while (i < (int)sizeof(int))
        {
            ptItem->signedValue |= (0xFF << ((i++) * 8));
        }
	}
	return kHIDSuccess;
}

