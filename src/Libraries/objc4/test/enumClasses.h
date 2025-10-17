/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#ifndef ENUMCLASSES_H_
#define ENUMCLASSES_H_

#include "test.h"

typedef enum {
    UnknownSize = -1,
    MinasculeSize,
    SmallSize,
    MediumSize,
    BigSize,
    HugeSize,
} creature_size_t;

typedef enum {
    BlackAndOrange,
    GrayAndBlack,
    Plaid,
} stripe_color_t;

@protocol Creature
- (const char *)name;
- (creature_size_t)size;
@end

@protocol Claws
- (void)retract;
- (void)extend;
@end

@protocol Stripes
- (stripe_color_t)stripeColor;
@end

// Animal
@interface Animal : TestRoot <Creature>

- (const char *)name;
- (creature_size_t)size;

@end

@interface Dog : Animal

- (const char *)name;

@end

@interface Cat : Animal

- (const char *)name;

@end

@interface Elephant : Animal

- (const char *)name;
- (creature_size_t)size;

@end

#endif /* ENUMCLASSES_H_ */
