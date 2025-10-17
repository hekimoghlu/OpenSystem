/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
//  MRCBase.h
//  TestARCLayouts
//
//  Created by Patrick Beard on 3/8/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <objc/NSObject.h>

// YES if MRC compiler supports ARC-style weak
extern bool supportsMRCWeak;

#if __LP64__
#define DOUBLEWORD_ALIGNED __attribute__((aligned(16)))
#else
#define DOUBLEWORD_ALIGNED __attribute__((aligned(8)))
#endif

@interface MRCBase : NSObject
@property double number;
@property(retain) id object;
@property void *pointer;
@property(weak) __weak id delegate;
@end

// Call object_copy from MRC.
extern id __attribute__((ns_returns_retained)) docopy(id obj);
