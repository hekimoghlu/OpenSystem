/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include "test.h"
#include <objc/runtime.h>

extern int state;

WEAK_IMPORT OBJC_ROOT_CLASS
@interface MissingRoot {
    id isa;
}
+(void) initialize;
+(Class) class;
+(id) alloc;
-(id) init;
-(void) dealloc;
+(int) method;
@end

@interface MissingRoot (RR)
-(id) retain;
-(void) release;
@end

WEAK_IMPORT
@interface MissingSuper : MissingRoot {
  @public
    int ivar;
}
@end

OBJC_ROOT_CLASS
@interface NotMissingRoot {
    id isa;
}
+(void) initialize;
+(Class) class;
+(id) alloc;
-(id) init;
-(void) dealloc;
+(int) method;
@end

@interface NotMissingRoot (RR)
-(id) retain;
-(void) release;
@end

@interface NotMissingSuper : NotMissingRoot {
  @public
    int unused[100];
    int ivar;
}
@end
