/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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

// TEST_CONFIG
/* TEST_BUILD_OUTPUT
.*designatedinit.m:\d+:\d+: warning: designated initializer should only invoke a designated initializer on 'super'.*
.*designatedinit.m:\d+:\d+: note: .*
.*designatedinit.m:\d+:\d+: warning: method override for the designated initializer of the superclass '-init' not found.*
.*NSObject.h:\d+:\d+: note: .*
END */

#define NS_ENFORCE_NSOBJECT_DESIGNATED_INITIALIZER 1
#include "test.h"
#include <objc/NSObject.h>

@interface C : NSObject
-(id) initWithInt:(int)i NS_DESIGNATED_INITIALIZER;
@end

@implementation C
-(id) initWithInt:(int)__unused i {
    return [self init];
}
@end

int main()
{
    succeed(__FILE__);
}
