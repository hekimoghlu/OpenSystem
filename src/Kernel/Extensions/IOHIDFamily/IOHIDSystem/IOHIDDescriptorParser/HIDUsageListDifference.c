/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
 * Date: Tuesday, May 23, 2023.
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
 * In - Is a usage in a UsageList?
 *
 *	 Input:
 *			  piUsageList			- usage List
 *			  iUsageListLength		- Max entries in usage Lists
 *			  usage				   - The usage
 *	 Output:
 *	 Returns: true or false
 *
 *------------------------------------------------------------------------------
*/
static Boolean IsUsageInUsageList(HIDUsage *piUsageList, UInt32 iUsageListLength, HIDUsage usage)
{
	unsigned int i;
    for (i = 0; i < iUsageListLength; i++) {
        if (piUsageList[i] == usage) {
			return true;
        }
    }
	return false;
}

/*
 *------------------------------------------------------------------------------
 *
 * HIDUsageListDifference - Return adds and drops given present and past
 *
 *	 Input:
 *			  piPreviouUL			- Previous usage List
 *			  piCurrentUL			- Current usage List
 *			  piBreakUL				- Break usage List
 *			  piMakeUL				- Make usage List
 *			  iUsageListLength		- Max entries in usage Lists
 *	 Output:
 *			  piBreakUL				- Break usage List
 *			  piMakeUL				- Make usage List
 *	 Returns:
 *
 *------------------------------------------------------------------------------
*/
OSStatus HIDUsageListDifference(HIDUsage *piPreviousUL, HIDUsage *piCurrentUL, HIDUsage *piBreakUL, HIDUsage *piMakeUL, UInt32 iUsageListLength)
{
	int i;
	HIDUsage usage;
	int iBreakLength=0;
	int iMakeLength=0;
	for (i = 0; i < (int)iUsageListLength; i++)
	{
/*
 *		If in Current but not Previous then it's a Make
*/
		usage = piCurrentUL[i];
		if ((usage != 0) && (!IsUsageInUsageList(piPreviousUL,iUsageListLength,usage))
						  && (!IsUsageInUsageList(piMakeUL,iMakeLength,usage)))
			piMakeUL[iMakeLength++] = usage;
/*
 *		If in Previous but not Current then it's a Break
*/
		usage = piPreviousUL[i];
		if ((usage != 0) && (!IsUsageInUsageList(piCurrentUL,iUsageListLength,usage))
						  && (!IsUsageInUsageList(piBreakUL,iBreakLength,usage)))
			piBreakUL[iBreakLength++] = usage;
	}
/*
 *	Clear the rest of the usage Lists
*/
	while (iMakeLength < (int)iUsageListLength)
		piMakeUL[iMakeLength++] = 0;
	while (iBreakLength < (int)iUsageListLength)
		piBreakUL[iBreakLength++] = 0;
	return kHIDSuccess;
}

