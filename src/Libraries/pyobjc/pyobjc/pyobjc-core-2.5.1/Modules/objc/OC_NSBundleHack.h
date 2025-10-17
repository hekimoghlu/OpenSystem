/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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
#import "pyobjc.h"
#import <Foundation/Foundation.h>

/*!
 * @class       OC_NSBundleHack
 * @abstract    NSBundle hacks to support plugins
 * @discussion
 *     This class that is used to post for NSBundle
 *     if it does not do the right thing
 */

@interface OC_NSBundleHack : NSBundle
{
}
+(void)installBundleHack;
@end

@interface OC_NSBundleHackCheck : NSObject
{
}
+(NSBundle*)bundleForClass;
@end
