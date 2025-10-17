/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
#import <Foundation/NSFileManager.h>

#if USE(APPLE_INTERNAL_SDK)

#import <Foundation/NSFileManager_NSURLExtras.h>

#else

#define WEB_UREAD (00400)
#define WEB_UWRITE (00200)
#define WEB_UEXEC (00100)

@interface NSFileManager ()
- (BOOL)_web_createDirectoryAtPathWithIntermediateDirectories:(NSString *)path attributes:(NSDictionary *)attributes;
- (BOOL)_web_createFileAtPath:(NSString *)path contents:(NSData *)contents attributes:(NSDictionary *)attributes;
- (BOOL)_web_removeFileOnlyAtPath:(NSString *)path;
@end

#endif
