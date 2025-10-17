/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
 * Date: Saturday, June 10, 2023.
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
 * HIDP_GetCaps
 *
 *	 Input:
 *			  ptPreparsedData		- Pre-Parsed Data
 *			  ptCapabilities		- Pointer to caller-provided structure
 *	 Output:
 *			  ptCapabilities		- Capabilities data
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetCaps(HIDPreparsedDataRef preparsedDataRef, HIDCapsPtr ptCapabilities)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollection *ptCollection;
	HIDReportItem *ptReportItem;
	HIDReportSizes *ptReport;
	int iFirstUsage;
	int i;
/*
 *	Disallow Null Pointers
*/

	if ((ptPreparsedData == NULL) || (ptCapabilities == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Copy the capabilities to the user
 *	Collection Capabilities
*/

	ptCollection = &ptPreparsedData->collections[1];
	ptCapabilities->usagePage = ptCollection->usagePage;
	iFirstUsage = ptCollection->firstUsageItem;
	ptCapabilities->usage = ptPreparsedData->usageItems[iFirstUsage].usage;
	ptCapabilities->numberCollectionNodes = ptPreparsedData->collectionCount;
/*
 *	Report Capabilities Summary
*/

	ptCapabilities->inputReportByteLength = 0;
	ptCapabilities->outputReportByteLength = 0;
	ptCapabilities->featureReportByteLength = 0;
	for (i=0; i< (int)ptPreparsedData->reportCount; i++)
	{
		ptReport = &ptPreparsedData->reports[i];
		if (ptCapabilities->inputReportByteLength < (IOByteCount)ptReport->inputBitCount)
			ptCapabilities->inputReportByteLength = ptReport->inputBitCount;
		if (ptCapabilities->outputReportByteLength < (IOByteCount)ptReport->outputBitCount)
			ptCapabilities->outputReportByteLength = ptReport->outputBitCount;
		if (ptCapabilities->featureReportByteLength < (IOByteCount)ptReport->featureBitCount)
			ptCapabilities->featureReportByteLength = ptReport->featureBitCount;
	}
	ptCapabilities->inputReportByteLength = (ptCapabilities->inputReportByteLength + 7) /8;
	ptCapabilities->outputReportByteLength = (ptCapabilities->outputReportByteLength + 7)/8;
	ptCapabilities->featureReportByteLength = (ptCapabilities->featureReportByteLength + 7)/8;
/*
 *	Sum the capabilities types
*/

	ptCapabilities->numberInputButtonCaps = 0;
	ptCapabilities->numberInputValueCaps = 0;
	ptCapabilities->numberOutputButtonCaps = 0;
	ptCapabilities->numberOutputValueCaps = 0;
	ptCapabilities->numberFeatureButtonCaps = 0;
	ptCapabilities->numberFeatureValueCaps = 0;
	for (i=0; i<(int)ptPreparsedData->reportItemCount; i++)
	{
		ptReportItem = &ptPreparsedData->reportItems[i];
		switch (ptReportItem->reportType)
		{
			case kHIDInputReport:
				if (HIDIsButton(ptReportItem, preparsedDataRef))
					ptCapabilities->numberInputButtonCaps += ptReportItem->usageItemCount;
				else if (HIDIsVariable(ptReportItem, preparsedDataRef))
					ptCapabilities->numberInputValueCaps += ptReportItem->usageItemCount;
				break;
			case kHIDOutputReport:
				if (HIDIsButton(ptReportItem, preparsedDataRef))
					ptCapabilities->numberOutputButtonCaps += ptReportItem->usageItemCount;
				else if (HIDIsVariable(ptReportItem, preparsedDataRef))
					ptCapabilities->numberOutputValueCaps += ptReportItem->usageItemCount;
				break;
			case kHIDFeatureReport:
				if (HIDIsButton(ptReportItem, preparsedDataRef))
					ptCapabilities->numberFeatureButtonCaps += ptReportItem->usageItemCount;
				else if (HIDIsVariable(ptReportItem, preparsedDataRef))
					ptCapabilities->numberFeatureValueCaps += ptReportItem->usageItemCount;
				break;
		}
	}
	return kHIDSuccess;
}


/*
 *------------------------------------------------------------------------------
 *
 * HIDGetCapabilities	This is exactly the same as HIDGetCaps. It does take a
 *						HIDCapabiitiesPtr instead of a HIDCapsPtr, but the structures
 *						of each are exactly the same. The only reason this call 
 *						exists seperately is for uniformity of naming with 
 *						HIDGetValueCapabilities, HIDGetSpecificButtonCapabilities, etc.
 *
 *	 Input:
 *			  ptPreparsedData		- Pre-Parsed Data
 *			  ptCapabilities		- Pointer to caller-provided structure
 *	 Output:
 *			  ptCapabilities		- Capabilities data
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetCapabilities(HIDPreparsedDataRef preparsedDataRef, HIDCapabilitiesPtr ptCapabilities)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollection *ptCollection;
	HIDReportItem *ptReportItem;
	HIDReportSizes *ptReport;
	int iFirstUsage;
	int i;
/*
 *	Disallow Null Pointers
*/

	if ((ptPreparsedData == NULL) || (ptCapabilities == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Copy the capabilities to the user
 *	Collection Capabilities
*/

	ptCollection = &ptPreparsedData->collections[1];
	ptCapabilities->usagePage = ptCollection->usagePage;
	iFirstUsage = ptCollection->firstUsageItem;
	ptCapabilities->usage = ptPreparsedData->usageItems[iFirstUsage].usage;
	ptCapabilities->numberCollectionNodes = ptPreparsedData->collectionCount;
/*
 *	Report Capabilities Summary
*/

	ptCapabilities->inputReportByteLength = 0;
	ptCapabilities->outputReportByteLength = 0;
	ptCapabilities->featureReportByteLength = 0;
	for (i=0; i<(int)ptPreparsedData->reportCount; i++)
	{
		ptReport = &ptPreparsedData->reports[i];
		if (ptCapabilities->inputReportByteLength < (IOByteCount)ptReport->inputBitCount)
			ptCapabilities->inputReportByteLength = ptReport->inputBitCount;
		if (ptCapabilities->outputReportByteLength < (IOByteCount)ptReport->outputBitCount)
			ptCapabilities->outputReportByteLength = ptReport->outputBitCount;
		if (ptCapabilities->featureReportByteLength < (IOByteCount)ptReport->featureBitCount)
			ptCapabilities->featureReportByteLength = ptReport->featureBitCount;
	}
	ptCapabilities->inputReportByteLength = (ptCapabilities->inputReportByteLength + 7) /8;
	ptCapabilities->outputReportByteLength = (ptCapabilities->outputReportByteLength + 7)/8;
	ptCapabilities->featureReportByteLength = (ptCapabilities->featureReportByteLength + 7)/8;
/*
 *	Sum the capabilities types
*/

	ptCapabilities->numberInputButtonCaps = 0;
	ptCapabilities->numberInputValueCaps = 0;
	ptCapabilities->numberOutputButtonCaps = 0;
	ptCapabilities->numberOutputValueCaps = 0;
	ptCapabilities->numberFeatureButtonCaps = 0;
	ptCapabilities->numberFeatureValueCaps = 0;
	for (i=0; i<(int)ptPreparsedData->reportItemCount; i++)
	{
		ptReportItem = &ptPreparsedData->reportItems[i];
		switch (ptReportItem->reportType)
		{
			case kHIDInputReport:
				if (HIDIsButton(ptReportItem, preparsedDataRef))
					ptCapabilities->numberInputButtonCaps += ptReportItem->usageItemCount;
				else if (HIDIsVariable(ptReportItem, preparsedDataRef))
					ptCapabilities->numberInputValueCaps += ptReportItem->usageItemCount;
				break;
			case kHIDOutputReport:
				if (HIDIsButton(ptReportItem, preparsedDataRef))
					ptCapabilities->numberOutputButtonCaps += ptReportItem->usageItemCount;
				else if (HIDIsVariable(ptReportItem, preparsedDataRef))
					ptCapabilities->numberOutputValueCaps += ptReportItem->usageItemCount;
				break;
			case kHIDFeatureReport:
				if (HIDIsButton(ptReportItem, preparsedDataRef))
					ptCapabilities->numberFeatureButtonCaps += ptReportItem->usageItemCount;
				else if (HIDIsVariable(ptReportItem, preparsedDataRef))
					ptCapabilities->numberFeatureValueCaps += ptReportItem->usageItemCount;
				break;
		}
	}
	return kHIDSuccess;
}

