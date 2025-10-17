/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 8, 2024.
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
#ifndef SecPLWrapper_h
#define SecPLWrapper_h


#if __OBJC__

#import <Foundation/Foundation.h>

extern void SecPLDisable(void);
extern bool SecPLShouldLogRegisteredEvent(NSString *event);

extern void SecPLLogRegisteredEvent(NSString *eventName, NSDictionary *eventDictionary);
extern void SecPLLogTimeSensitiveRegisteredEvent(NSString *eventName, NSDictionary *eventDictionary);

#else

#include <CoreFoundation/CoreFoundation.h>

extern void SecPLDisable(void);
extern bool SecPLShouldLogRegisteredEvent(CFStringRef event);

extern void SecPLLogRegisteredEvent(CFStringRef eventName, CFDictionaryRef eventDictionary);
extern void SecPLLogTimeSensitiveRegisteredEvent(CFStringRef eventName, CFDictionaryRef eventDictionary);

#endif

#endif /* SecPLWrapper_h */
