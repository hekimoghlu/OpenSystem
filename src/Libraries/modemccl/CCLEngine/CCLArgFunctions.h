/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
 *  CCLArgFunctions.h
 *  CCLLauncher
 *
 *  Created by kevine on 4/11/06.
 *  Copyright 2006-7 Apple, Inc.  All rights reserved.
 *
 */

#ifndef __CCLArgFunctions__
#define __CCLArgFunctions__

#include <CoreFoundation/CoreFoundation.h>


#define kCCL_BundleExtension    "ccl"
#define kCCL_BundleExtLen       sizeof(kCCL_BundleExtension) - sizeof('\0') 

CFDictionaryRef
GetCFDictionaryFromDict(CFDictionaryRef dict, const CFStringRef key);

bool
GetCFStringFromDict(CFDictionaryRef dict, CFStringRef *s,const CFStringRef key);
bool
CopyCStringFromDict(CFDictionaryRef dict, char** string, const CFStringRef key);

bool
GetCFNumberFromDict(CFDictionaryRef dict, CFNumberRef *n,const CFStringRef key);
bool
GetIntFromDict(CFDictionaryRef dict, int* intRef, const CFStringRef key);


#endif  // __CCLArgFunctions__
