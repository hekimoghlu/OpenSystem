/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
 * Date: Tuesday, November 2, 2021.
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
 * HIDGetSpecificButtonCaps - Get the binary values for a report type
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage			   - Page Criteria or zero
 *			  iCollection			- Collection Criteria or zero
 *			  usage				   - usage Criteria or zero
 *			  buttonCaps		  - ButtonCaps Array
 *			  piButtonCapsLength	- Maximum Entries
 *			  ptPreparsedData		- Pre-Parsed Data
 *	 Output:
 *			  piButtonCapsLength	- Entries Populated
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetSpecificButtonCaps(HIDReportType reportType,
									   HIDUsage usagePage,
									   UInt32 iCollection,
									   HIDUsage usage,
									   HIDButtonCapsPtr buttonCaps,
									   UInt32 *piButtonCapsLength,
									   HIDPreparsedDataRef preparsedDataRef)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollection *ptCollection;
	HIDCollection *ptParent;
	HIDReportItem *ptReportItem;
	HIDP_UsageItem *ptUsageItem;
	HIDStringItem *ptStringItem;
	HIDDesignatorItem *ptDesignatorItem;
	HIDP_UsageItem *ptFirstCollectionUsageItem;
	HIDButtonCaps *ptCapability;
	int iR, iU;
	int parent;
	int iReportItem, iUsageItem;
	int iMaxCaps;
		// There are 3 versions of HID Parser code all based on the same logic: OS 9 HID Library;
		// OSX xnu; OSX IOKitUser. They should all be nearly the same logic. This version (xnu)
		// is based on older OS 9 code. This version has added logic to maintain this startBit.
		// I don't know why it is here, but believe if it is needed here, it would probably be
		// needed in the other two implementations. Didn't have time to determine that at this 
		// time, so i'll leave this comment to remind me that we should reconcile the 3 versions.
        UInt32 startBit;	// Added esb 9-29-99
	/*If I remember correctly, it was an optimization.  Each time you ask for 
	a specific value capability, it would search through the entire report 
	descriptor to find it (my recollection is kind of hazy on this part).  
	The start bit allowed somebody (client maybe) to cache the information 
	on where in the report a specific value resided and the use that later 
	when fetching that value.  That way, you don't have to keep going 
	through the parse tree to find where a value exists.  I don't remember 
	if the implementation was completed or if I even used it. -esb */
/*
 *	Disallow Null Pointers
*/
	if ((buttonCaps == NULL)
	 || (piButtonCapsLength == NULL)
	 || (ptPreparsedData == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Save the buffer size
*/
	iMaxCaps = *piButtonCapsLength;
	*piButtonCapsLength = 0;
/*
 *	The Collection must be in range
*/
	if (iCollection >= ptPreparsedData->collectionCount)
		return kHIDBadParameterErr;
/*
 *	Search only the scope of the Collection specified
*/
	ptCollection = &ptPreparsedData->collections[iCollection];
	for (iR=0; iR<ptCollection->reportItemCount; iR++)
	{
		iReportItem = ptCollection->firstReportItem + iR;
		ptReportItem = &ptPreparsedData->reportItems[iReportItem];
/*
 *		Search only reports of the proper type
*/
		if ((ptReportItem->reportType == reportType)
		 && HIDIsButton(ptReportItem, preparsedDataRef))
		{
                        startBit = ptReportItem->startBit;
/*
 *			Search the usages
*/
			  for (iU=0; iU<ptReportItem->usageItemCount; iU++)
			  {
/*
 *				  Copy all usages if the usage above is zero
 *					or copy all that are "match"
*/
				  iUsageItem = ptReportItem->firstUsageItem + iU;
				  ptUsageItem = &ptPreparsedData->usageItems[iUsageItem];

				  // Â¥Â¥ we assume there is a 1-1 corresponence between usage items, string items, and designator items
				  // Â¥Â¥ÃŠthis is not necessarily the case, but its better than nothing
				  ptStringItem = &ptPreparsedData->stringItems[ptReportItem->firstStringItem + iU];
				  ptDesignatorItem = &ptPreparsedData->desigItems[ptReportItem->firstDesigItem + iU];

				  if (HIDUsageInRange(ptUsageItem,usagePage,usage))
				  {
/*
 *					  Only copy if there's room
*/
					  if (*piButtonCapsLength >= (UInt32)iMaxCaps)
						  return kHIDBufferTooSmallErr;
					  ptCapability = &buttonCaps[(*piButtonCapsLength)++];
/*
 *					  Populate the Capability Structure
*/
					  parent = ptReportItem->parent;
					  ptParent = &ptPreparsedData->collections[parent];
					  ptFirstCollectionUsageItem
						 = &ptPreparsedData->usageItems[ptParent->firstUsageItem];
					  ptCapability->collection = parent;
					  ptCapability->collectionUsagePage = ptParent->usagePage;
					  ptCapability->collectionUsage = ptFirstCollectionUsageItem->usage;
					  ptCapability->bitField =	ptReportItem->dataModes;
					  ptCapability->reportID = ptReportItem->globals.reportID;
					  ptCapability->usagePage = ptUsageItem->usagePage;
					  
					  ptCapability->isStringRange = false;			// Â¥Â¥ todo: set this and stringMin,stringMax,stringIndex
					  ptCapability->isDesignatorRange = false;		// Â¥Â¥ todo: set this and designatorMin,designatorMax,designatorIndex
					  ptCapability->isAbsolute = !(ptReportItem->dataModes & kHIDDataRelative);

					  ptCapability->isRange = ptUsageItem->isRange;
					  if (ptUsageItem->isRange)
					  {
						ptCapability->u.range.usageMin = ptUsageItem->usageMinimum;
						ptCapability->u.range.usageMax = ptUsageItem->usageMaximum;
					  }
					  else
						ptCapability->u.notRange.usage = ptUsageItem->usage;

					  // if there really are that many items
					  if (iU < ptReportItem->stringItemCount)
					  {
						  ptCapability->isStringRange = ptStringItem->isRange;
						  
						  if (ptStringItem->isRange)
						  {
							ptCapability->u.range.stringMin = ptStringItem->minimum;
							ptCapability->u.range.stringMax = ptStringItem->maximum;
						  }
						  else
							ptCapability->u.notRange.stringIndex = ptStringItem->index;
					  }
					  // default, clear it
					  else
					  {
					  	ptCapability->isStringRange = false;
						ptCapability->u.notRange.stringIndex = 0;
					  }

					  // if there really are that many items
					  if (iU < ptReportItem->desigItemCount)
					  {
						  ptCapability->isDesignatorRange = ptDesignatorItem->isRange;
						  
						  if (ptDesignatorItem->isRange)
						  {
							ptCapability->u.range.designatorMin = ptDesignatorItem->minimum;
							ptCapability->u.range.designatorMax = ptDesignatorItem->maximum;
						  }
						  else
							ptCapability->u.notRange.designatorIndex = ptDesignatorItem->index;
					  }
					  // default, clear it
					  else
					  {
					  	ptCapability->isDesignatorRange = false;
						ptCapability->u.notRange.designatorIndex = 0;
					  }
                                          ptCapability->startBit = startBit;
				  }
                                  if ((ptReportItem->dataModes & kHIDDataVariableBit) == kHIDDataVariable)
                                  {
                                      // RY: Incremeneting start bit too much in case or multiple usage items
                                      //startBit += (ptReportItem->globals.reportSize * ptReportItem->globals.reportCount);
                                      startBit += ptReportItem->globals.reportSize;
                                  }
			  }
		}
	}
	return kHIDSuccess;
}

/*
 *------------------------------------------------------------------------------
 *
 * HIDGetButtonCaps - Get the binary values for a report type
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  buttonCaps		  - ButtonCaps Array
 *			  piButtonCapsLength	- Maximum Entries
 *			  ptPreparsedData		- Pre-Parsed Data
 *	 Output:
 *			  piButtonCapsLength	- Entries Populated
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetButtonCaps(HIDReportType reportType,
							   HIDButtonCapsPtr buttonCaps,
							   UInt32 *piButtonCapsLength,
							   HIDPreparsedDataRef preparsedDataRef)
{
	return HIDGetSpecificButtonCaps(reportType,0,0,0,buttonCaps,
									  piButtonCapsLength,preparsedDataRef);
}


/*
 *------------------------------------------------------------------------------
 *
 * HIDGetSpecificButtonCapabilities - Get the binary values for a report type
 *								This is the same as HIDGetSpecificButtonCaps,
 *								except that it takes a HIDButtonCapabilitiesPtr
 *								so it can return units and unitExponents.
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  usagePage			   - Page Criteria or zero
 *			  iCollection			- Collection Criteria or zero
 *			  usage				   - usage Criteria or zero
 *			  buttonCaps		  - ButtonCaps Array
 *			  piButtonCapsLength	- Maximum Entries
 *			  ptPreparsedData		- Pre-Parsed Data
 *	 Output:
 *			  piButtonCapsLength	- Entries Populated
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetSpecificButtonCapabilities(HIDReportType reportType,
									   HIDUsage usagePage,
									   UInt32 iCollection,
									   HIDUsage usage,
									   HIDButtonCapabilitiesPtr buttonCaps,
									   UInt32 *piButtonCapsLength,
									   HIDPreparsedDataRef preparsedDataRef)
{
	HIDPreparsedDataPtr ptPreparsedData = (HIDPreparsedDataPtr) preparsedDataRef;
	HIDCollection *ptCollection;
	HIDCollection *ptParent;
	HIDReportItem *ptReportItem;
	HIDP_UsageItem *ptUsageItem;
	HIDStringItem *ptStringItem;
	HIDDesignatorItem *ptDesignatorItem;
	HIDP_UsageItem *ptFirstCollectionUsageItem;
	HIDButtonCapabilities *ptCapability;
	int iR, iU;
	int parent;
	int iReportItem, iUsageItem;
	int iMaxCaps;
		// There are 3 versions of HID Parser code all based on the same logic: OS 9 HID Library;
		// OSX xnu; OSX IOKitUser. They should all be nearly the same logic. This version (xnu)
		// is based on older OS 9 code. This version has added logic to maintain this startBit.
		// I don't know why it is here, but believe if it is needed here, it would probably be
		// needed in the other two implementations. Didn't have time to determine that at this 
		// time, so i'll leave this comment to remind me that we should reconcile the 3 versions.
        UInt32 startBit;
/*
 *	Disallow Null Pointers
*/
	if ((buttonCaps == NULL)
	 || (piButtonCapsLength == NULL)
	 || (ptPreparsedData == NULL))
		return kHIDNullPointerErr;
	if (ptPreparsedData->hidTypeIfValid != kHIDOSType)
		return kHIDInvalidPreparsedDataErr;
/*
 *	Save the buffer size
*/
	iMaxCaps = *piButtonCapsLength;
	*piButtonCapsLength = 0;
/*
 *	The Collection must be in range
*/
	if (iCollection >= ptPreparsedData->collectionCount)
		return kHIDBadParameterErr;
/*
 *	Search only the scope of the Collection specified
*/
	ptCollection = &ptPreparsedData->collections[iCollection];
	for (iR=0; iR<ptCollection->reportItemCount; iR++)
	{
		iReportItem = ptCollection->firstReportItem + iR;
		ptReportItem = &ptPreparsedData->reportItems[iReportItem];
/*
 *		Search only reports of the proper type
*/
		if ((ptReportItem->reportType == reportType)
		 && HIDIsButton(ptReportItem, preparsedDataRef))
		{
                        startBit = ptReportItem->startBit;
/*
 *			Search the usages
*/
			  for (iU=0; iU<ptReportItem->usageItemCount; iU++)
			  {
/*
 *				  Copy all usages if the usage above is zero
 *					or copy all that are "match"
*/
				  iUsageItem = ptReportItem->firstUsageItem + iU;
				  ptUsageItem = &ptPreparsedData->usageItems[iUsageItem];

				  // Â¥Â¥ we assume there is a 1-1 corresponence between usage items, string items, and designator items
				  // Â¥Â¥ÃŠthis is not necessarily the case, but its better than nothing
				  ptStringItem = &ptPreparsedData->stringItems[ptReportItem->firstStringItem + iU];
				  ptDesignatorItem = &ptPreparsedData->desigItems[ptReportItem->firstDesigItem + iU];

				  if (HIDUsageInRange(ptUsageItem,usagePage,usage))
				  {
/*
 *					  Only copy if there's room
*/
					  if (*piButtonCapsLength >= (UInt32)iMaxCaps)
						  return kHIDBufferTooSmallErr;
					  ptCapability = &buttonCaps[(*piButtonCapsLength)++];
/*
 *					  Populate the Capability Structure
*/
					  parent = ptReportItem->parent;
					  ptParent = &ptPreparsedData->collections[parent];
					  ptFirstCollectionUsageItem
						 = &ptPreparsedData->usageItems[ptParent->firstUsageItem];
					  ptCapability->collection = parent;
					  ptCapability->collectionUsagePage = ptParent->usagePage;
					  ptCapability->collectionUsage = ptFirstCollectionUsageItem->usage;
					  ptCapability->bitField =	ptReportItem->dataModes;
					  ptCapability->reportID = ptReportItem->globals.reportID;
					  ptCapability->usagePage = ptUsageItem->usagePage;
                                          
                                          // *** HACK FOR ARRAY ITEMS ***
                                          if ((ptReportItem->dataModes & kHIDDataArrayBit) == kHIDDataArray)
                                          {
                                            // RY: Introducing a hack to allow the HID Manager
                                            // gain access to the report Size and report Count
                                            // for array items.  Trust me this makes sense
                                            // becuase these fields are not used by an array.
                                            ptCapability->unitExponent = (SInt32)ptReportItem->globals.reportSize;
                                            ptCapability->units = ptReportItem->globals.reportCount;
                                            
                                            // RY: For array items, we need to know the logical
                                            // min/max.  Since we don't have any space, we need
                                            // to use 2 reserved fields of the non usage union
                                            
                                            if (ptReportItem->flags & kHIDReportItemFlag_Reversed)
                                            {
                                                ptCapability->u.notRange.reserved2 = ptReportItem->globals.logicalMaximum;
                                                ptCapability->u.notRange.reserved3 = ptReportItem->globals.logicalMinimum;
                                            }
                                            else 
                                            {
                                                ptCapability->u.notRange.reserved2 = ptReportItem->globals.logicalMinimum;
                                                ptCapability->u.notRange.reserved3 = ptReportItem->globals.logicalMaximum;
                                            }

                                          }
                                          else
                                          {
                                            ptCapability->unitExponent = ptReportItem->globals.unitExponent;
                                            ptCapability->units = ptReportItem->globals.units;
                                          }
                                          // End hack
                                          
//					  ptCapability->reserved = 0;							// for future OS 9 expansion
					  ptCapability->startBit = 0;		// init esb added field.
//					  ptCapability->pbVersion = kHIDCurrentCapabilitiesPBVersion;
					  ptCapability->pbVersion = 2;
					  
					  ptCapability->isStringRange = false;			// Â¥Â¥ todo: set this and stringMin,stringMax,stringIndex
					  ptCapability->isDesignatorRange = false;		// Â¥Â¥ todo: set this and designatorMin,designatorMax,designatorIndex
					  ptCapability->isAbsolute = !(ptReportItem->dataModes & kHIDDataRelative);

					  ptCapability->isRange = ptUsageItem->isRange;
					  if (ptUsageItem->isRange)
					  {
						ptCapability->u.range.usageMin = ptUsageItem->usageMinimum;
						ptCapability->u.range.usageMax = ptUsageItem->usageMaximum;
					  }
					  else
						ptCapability->u.notRange.usage = ptUsageItem->usage;

					  // if there really are that many items
					  if (iU < ptReportItem->stringItemCount)
					  {
						  ptCapability->isStringRange = ptStringItem->isRange;
						  
						  if (ptStringItem->isRange)
						  {
							ptCapability->u.range.stringMin = ptStringItem->minimum;
							ptCapability->u.range.stringMax = ptStringItem->maximum;
						  }
						  else
							ptCapability->u.notRange.stringIndex = ptStringItem->index;
					  }
					  // default, clear it
					  else
					  {
					  	ptCapability->isStringRange = false;
						ptCapability->u.notRange.stringIndex = 0;
					  }

					  // if there really are that many items
					  if (iU < ptReportItem->desigItemCount)
					  {
						  ptCapability->isDesignatorRange = ptDesignatorItem->isRange;
						  
						  if (ptDesignatorItem->isRange)
						  {
							ptCapability->u.range.designatorMin = ptDesignatorItem->minimum;
							ptCapability->u.range.designatorMax = ptDesignatorItem->maximum;
						  }
						  else
							ptCapability->u.notRange.designatorIndex = ptDesignatorItem->index;
					  }
					  // default, clear it
					  else
					  {
					  	ptCapability->isDesignatorRange = false;
						ptCapability->u.notRange.designatorIndex = 0;
					  }
                                          ptCapability->startBit = startBit;
				  }
                                  // For array items, we want the startBit left alone;
                                  if ((ptReportItem->dataModes & kHIDDataVariableBit) == kHIDDataVariable)
                                  {
                                    // RY: Incremeneting start bit too much in case or multiple usage items
                                    //startBit += (ptReportItem->globals.reportSize * ptReportItem->globals.reportCount);
                                    startBit += ptReportItem->globals.reportSize;
                                  }
			  }
		}
	}
	return kHIDSuccess;
}

/*
 *------------------------------------------------------------------------------
 *
 * HIDGetButtonCapabilities - Get the binary values for a report type
 *								This is the same as HIDGetButtonCaps,
 *								except that it takes a HIDButtonCapabilitiesPtr
 *								so it can return units and unitExponents.
 *
 *	 Input:
 *			  reportType		   - HIDP_Input, HIDP_Output, HIDP_Feature
 *			  buttonCaps		  - ButtonCaps Array
 *			  piButtonCapsLength	- Maximum Entries
 *			  ptPreparsedData		- Pre-Parsed Data
 *	 Output:
 *			  piButtonCapsLength	- Entries Populated
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDGetButtonCapabilities(HIDReportType reportType,
							   HIDButtonCapabilitiesPtr buttonCaps,
							   UInt32 *piButtonCapsLength,
							   HIDPreparsedDataRef preparsedDataRef)
{
	return HIDGetSpecificButtonCapabilities(reportType,0,0,0,buttonCaps,
									  piButtonCapsLength,preparsedDataRef);
}

