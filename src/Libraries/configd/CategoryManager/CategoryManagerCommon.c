/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
 * CategoryManagerCommon.c
 */

/*
 * Modification History
 *
 * December 22, 2022	Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

#include "CategoryManagerCommon.h"
#include "CategoryManagerInternal.h"
#include <SystemConfiguration/SCValidation.h>
#include "symbol_scope.h"

PRIVATE_EXTERN CFStringRef
CategoryInformationGetCategory(CategoryInformationRef info)
{
	CFStringRef	str;

	str = CFDictionaryGetValue(info, kCategoryInformationKeyCategory);
	return (isA_CFString(str));
}

PRIVATE_EXTERN CFStringRef
CategoryInformationGetInterfaceName(CategoryInformationRef info)
{
	CFStringRef	str;

	str = CFDictionaryGetValue(info, kCategoryInformationKeyInterfaceName);
	return (isA_CFString(str));
}

PRIVATE_EXTERN CFStringRef
CategoryInformationGetValue(CategoryInformationRef info)
{
	CFStringRef	str;

	str = CFDictionaryGetValue(info, kCategoryInformationKeyValue);
	return (isA_CFString(str));
}

PRIVATE_EXTERN SCNetworkCategoryManagerFlags
CategoryInformationGetFlags(CategoryInformationRef info)
{
	SCNetworkCategoryManagerFlags	flags = 0;
	CFNumberRef			flags_cf;

	flags_cf = CFDictionaryGetValue(info, kCategoryInformationKeyFlags);
	if (isA_CFNumber(flags_cf) != NULL) {
		CFNumberGetValue(flags_cf, kCFNumberSInt32Type, &flags);
	}
	return (flags);
}

