/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 16, 2024.
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
#ifndef _CATEGORY_MANAGER_COMMON_H
#define _CATEGORY_MANAGER_COMMON_H

/*
 * CategoryManagerCommon.h
 */

/*
 * Modification History
 *
 * December 22, 2022	Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

#include <CoreFoundation/CoreFoundation.h>
#include "SCNetworkCategoryManager.h"

typedef CFDictionaryRef CategoryInformationRef;

CFStringRef
CategoryInformationGetCategory(CategoryInformationRef info);

CFStringRef
CategoryInformationGetValue(CategoryInformationRef info);

CFStringRef
CategoryInformationGetInterfaceName(CategoryInformationRef info);

SCNetworkCategoryManagerFlags
CategoryInformationGetFlags(CategoryInformationRef info);

#endif /* _CATEGORY_MANAGER_SERVER_H */
