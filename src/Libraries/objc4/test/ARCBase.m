/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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
//  ARCBase.m
//  TestARCLayouts
//
//  Created by Patrick Beard on 3/8/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "ARCBase.h"

// ARCMisalign->misalign1 and ARCBase->misalign2 together cause 
// ARCBase's instanceStart to be misaligned, which exercises handling 
// of storage that is not represented in the class's ivar layout bitmaps.

@implementation ARCMisalign
@end

@interface ARCBase () {
@private
    char misalign2;
    long number;
    id object;
    void *pointer;
    __weak id delegate;
}
@end

@implementation ARCBase
@synthesize number, object, pointer, delegate;
@end
