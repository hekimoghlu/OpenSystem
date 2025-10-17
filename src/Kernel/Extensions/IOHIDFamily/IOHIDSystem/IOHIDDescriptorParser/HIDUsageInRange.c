/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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
 * Date: Tuesday, February 18, 2025.
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
 * HIDUsageInRange
 *
 *	 Input:
 *			  ptUsage				- The usage/UsageRange Item
 *			  usagePage			   - The usagePage of the Item - or zero
 *			  usage				   - The usage of the Item
 *	 Output:
 *	 Returns:
 *			  true					- usagePage/usage is in usage/UsageRange
 *			  false					- usagePage/usage is not in usage/UsageRange
 *
 *------------------------------------------------------------------------------
*/
Boolean HIDUsageInRange (HIDP_UsageItem *ptUsage, HIDUsage usagePage, HIDUsage usage)
{
/*
 *	Disallow Null Pointers
*/
	if (ptUsage == NULL)
		return false;
/*
 *	Check for the proper Page, 0 means don't care
*/
	if ((usagePage != 0) && (ptUsage->usagePage != usagePage))
		return false;
/*
 *	usage = 0 means don't care
*/
	if (usage == 0)
		return true;
/*
 *	The requested usage must match or be in the range
*/
	if (ptUsage->isRange)
	{
		if ((ptUsage->usageMinimum > (SInt32)usage) || (ptUsage->usageMaximum < (SInt32)usage))
			return false;
	}
	else
	{
		if (ptUsage->usage != usage)
			return false;
	}
	return true;
}

