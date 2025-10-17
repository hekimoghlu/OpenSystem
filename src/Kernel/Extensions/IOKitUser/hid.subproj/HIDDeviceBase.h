/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
//  HIDDeviceBase.h
//  iohidobjc
//
//  Created by dekom on 10/17/18.
//

#ifndef HIDDeviceBase_h
#define HIDDeviceBase_h

#if __OBJC__

#import <IOKit/hidobjc/HIDDeviceIvar.h>
#import <CoreFoundation/CoreFoundation.h>
#import <objc/NSObject.h>

@interface HIDDevice : NSObject {
@protected
    HIDDeviceStruct _device;
}

@end

#endif /* __OBJC__ */

#endif /* HIDDeviceBase_h */
