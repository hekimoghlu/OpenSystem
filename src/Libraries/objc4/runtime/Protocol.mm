/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
/*
	Protocol.h
	Copyright 1991-1996 NeXT Software, Inc.
*/

#include "objc-private.h"

#undef id
#undef Class

#include <stdlib.h>
#include <string.h>
#include <mach-o/dyld.h>

#include "Protocol.h"
#include "NSObject.h"

// __IncompleteProtocol is used as the return type of objc_allocateProtocol().

// Old ABI uses NSObject as the superclass even though Protocol uses Object
// because the R/R implementation for class Protocol is added at runtime
// by CF, so __IncompleteProtocol would be left without an R/R implementation 
// otherwise, which would break ARC.

@interface __IncompleteProtocol : NSObject
@end

__attribute__((objc_nonlazy_class))
@implementation __IncompleteProtocol
@end


__attribute__((objc_nonlazy_class))
@implementation Protocol

- (BOOL) conformsTo: (Protocol *)aProtocolObj
{
    return protocol_conformsToProtocol(self, aProtocolObj);
}

- (struct objc_method_description *) descriptionForInstanceMethod:(SEL)aSel
{
    return method_getDescription(protocol_getMethod((struct protocol_t *)self, 
                                                     aSel, YES, YES, YES));
}

- (struct objc_method_description *) descriptionForClassMethod:(SEL)aSel
{
    return method_getDescription(protocol_getMethod((struct protocol_t *)self, 
                                                    aSel, YES, NO, YES));
}

- (const char *)name
{
    return protocol_getName(self);
}

- (BOOL)isEqual:other
{
    // check isKindOf:
    Class cls;
    Class protoClass = objc_getClass("Protocol");
    for (cls = object_getClass(other); cls; cls = cls->getSuperclass()) {
        if (cls == protoClass) break;
    }
    if (!cls) return NO;
    // check equality
    return protocol_isEqual(self, other);
}

- (NSUInteger)hash
{
    return 23;
}

@end
