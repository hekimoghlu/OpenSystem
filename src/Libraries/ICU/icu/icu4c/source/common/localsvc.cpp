/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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

//
// localsvc.cpp -- Apple-specific service hook
// Copyright (c) 2007 Apple Inc. All rights reserved.
//
// originally added per rdar://4448220 Add user dictionary support
//

#include "unicode/utypes.h"

#if !UCONFIG_NO_BREAK_ITERATION

#include "aaplbfct.h"
#include "cstring.h"
// platform.h now includes <TargetConditionals.h> if U_PLATFORM_IS_DARWIN_BASED

// Return an appropriate Apple-specific object, based on the service in question
U_CAPI void* uprv_svc_hook(const char *what, UErrorCode *status)
{
	if (uprv_strcmp(what, "languageBreakFactory") == 0) {
// NOTE: The implementation of AppleLanguageBreakFactory has been commented out since 2012 (check git blame on
// aaplbfct.cpp), and having it here interferes with the new external-break-engine stuff in ICU 74, so comment
// it out here.  If we need our own language break engine in the future, we need to use the new mechanism.
#if /*U_PLATFORM_IS_DARWIN_BASED && TARGET_OS_MAC*/ 0
		return new icu::AppleLanguageBreakFactory(*status);
	}
#else
	}
#endif
	return NULL;
}

#endif
