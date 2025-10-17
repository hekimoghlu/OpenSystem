/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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
//  MRCBase.m
//  TestARCLayouts
//
//  Created by Patrick Beard on 3/8/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#include "MRCBase.h"
#include "test.h"

// MRCBase->alignment ensures that there is a gap between the end of 
// NSObject's ivars and the start of MRCBase's ivars, which exercises 
// handling of storage that is not represented in any class's ivar 
// layout bitmaps.

#if __has_feature(objc_arc_weak)
bool supportsMRCWeak = true;
#else
bool supportsMRCWeak = false;
#endif

@interface MRCBase () {
@private
    double DOUBLEWORD_ALIGNED alignment;
    uintptr_t pad[3]; // historically this made OBJC2 layout bitmaps match OBJC1
    double number;
    id object;
    void *pointer;
#if __has_feature(objc_arc_weak)
    __weak 
#endif
    id delegate;
}
@end

@implementation MRCBase
@synthesize number, object, pointer, delegate;
@end

// Call object_copy from MRC.
extern id __attribute__((ns_returns_retained)) 
docopy(id obj)
{
    return object_copy(obj, 0);
}
