/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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
 * Date: Sunday, August 18, 2024.
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
 *------------------------------------------------------------------------------
 *
 * HIDMaxUsageListLength
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage			   - Page Criteria or zero
 *			  ptPreparsedData		- Pre-Parsed Data
 *	 Output:
 *	 Returns: length of list
 *
 *------------------------------------------------------------------------------
*/
UInt32
HIDMaxUsageListLength	   (HIDReportType reportType,
							HIDUsage usagePage,
							HIDPreparsedDataRef preparsedDataRef)
{
#pragma unused(usagePage)	// not used, see comment below

	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDReportItem *ptReportItem;
	int iButtons;
	int i;

	
/*
 *	Disallow Null Pointers
*/
	if (ptPreparsedData == NULL)
		return 0;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Go through the ReportItems
 *	Filter on ReportType
 *	Sum the button counts
 *
 * NOTE: A more precise value for the maximum list length
 *		 may be obtained by filtering out the usages that
 *		 are not on the specified usage page.  Most of
 *		 the time the number returned below is the same
 *		 as that returned by filtering usages.	It is
 *		 never smaller.	 The tradeoff is sometimes wasting
 *		 a few words of RAM in exchange for speed.
*/
	iButtons = 0;
	for (i=0; i< (int)ptPreparsedData->reportItemCount; i++)
	{
		ptReportItem = &ptPreparsedData->reportItems[i];
		if ((ptReportItem->reportType == reportType)
		 && HIDIsButton(ptReportItem, preparsedDataRef))
			iButtons += ptReportItem->globals.reportCount;
	}
	return iButtons;
}

