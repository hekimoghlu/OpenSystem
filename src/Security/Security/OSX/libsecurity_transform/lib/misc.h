/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
#ifndef __INCLUDED_TRANSFORMS_MISC_H__
#define __INCLUDED_TRANSFORMS_MISC_H__

#include <stdio.h>
#include "SecTransform.h"

#ifdef __cplusplus
extern "C" {
#endif
	
	
	CFErrorRef fancy_error(CFStringRef domain, CFIndex code, CFStringRef description);
	extern void CFfprintf(FILE *f, CFStringRef format, ...) __attribute__((format(__CFString__, 2, 3)));
	CFErrorRef GetNoMemoryError(void);
	CFErrorRef GetNoMemoryErrorAndRetain(void);
	void CFSafeRelease(CFTypeRef object);
    
    // NOTE: the return may or allocate a fair bit more space then it needs.
    // Use it for short lived conversions (or strdup the result).
    extern char *utf8(CFStringRef s);

#ifdef __cplusplus
}
#endif
		
#endif
