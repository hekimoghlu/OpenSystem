/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 29, 2024.
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
#include "testroot.i"
#include <stdint.h>
#include <string.h>
#include <objc/objc-runtime.h>


@protocol SuperProto
@property(readonly) char superProtoProp;
@property(class,readonly) char superProtoProp;
@end

@protocol SubProto <SuperProto>
@property(readonly) uintptr_t subProtoProp;
@property(class,readonly) uintptr_t subProtoProp;
@property(readonly) uintptr_t subInstanceOnlyProtoProp;
@property(class,readonly) uintptr_t subClassOnlyProtoProp;
@end

@interface Super : TestRoot <SuperProto> { 
  @public
    char superIvar;
}

@property(readonly) char superProp;
@property(class,readonly) char superProp;
@end

@implementation Super
@synthesize superProp = superIvar;
+(char)superProp { return 'a'; }

-(char)superProtoProp { return 'a'; }
+(char)superProtoProp { return 'a'; }
@end


@interface Sub : Super <SubProto> {
  @public 
    uintptr_t subIvar;
}
@property(readonly) uintptr_t subProp;
@property(class,readonly) uintptr_t subProp;
@property(readonly) uintptr_t subInstanceOnlyProp;
@property(class,readonly) uintptr_t subClassOnlyProp;
@end

@implementation Sub 
@synthesize subProp = subIvar;
+(uintptr_t)subProp { return 'a'; }
+(uintptr_t)subClassOnlyProp { return 'a'; }
-(uintptr_t)subInstanceOnlyProp { return 'a'; }

-(uintptr_t)subProtoProp { return 'a'; }
+(uintptr_t)subProtoProp { return 'a'; }
+(uintptr_t)subClassOnlyProtoProp { return 'a'; }
-(uintptr_t)subInstanceOnlyProtoProp { return 'a'; }
@end


void test(Class subcls) 
{
    objc_property_t prop;

    Class supercls = class_getSuperclass(subcls);

    prop = class_getProperty(subcls, "subProp");
    testassert(prop);

    prop = class_getProperty(subcls, "subProtoProp");
    testassert(prop);

    prop = class_getProperty(supercls, "superProp");
    testassert(prop);
    testassert(prop == class_getProperty(subcls, "superProp"));

    prop = class_getProperty(supercls, "superProtoProp");
    testassert(prop);
    // These are distinct because Sub adopts SuperProto itself 
    // in addition to Super's adoption of SuperProto.
    testassert(prop != class_getProperty(subcls, "superProtoProp"));

    prop = class_getProperty(supercls, "subProp");
    testassert(!prop);

    prop = class_getProperty(supercls, "subProtoProp");
    testassert(!prop);

    testassert(nil == class_getProperty(nil, "foo"));
    testassert(nil == class_getProperty(subcls, nil));
    testassert(nil == class_getProperty(nil, nil));
}


int main()
{    
    Class subcls = [Sub class];
    Class submeta = object_getClass(subcls);
    objc_property_t prop;

    // instance properties
    test(subcls);

    // class properties
    test(submeta);

    // properties must not appear on the wrong side
    testassert(nil == class_getProperty(subcls, "subClassOnlyProp"));
    testassert(nil == class_getProperty(submeta, "subInstanceOnlyProp"));
    testassert(nil == class_getProperty(subcls, "subClassOnlyProtoProp"));
    testassert(nil == class_getProperty(submeta, "subInstanceOnlyProtoProp"));

    // properties with the same name on both sides are distinct
    testassert(class_getProperty(subcls, "subProp") != class_getProperty(submeta, "subProp"));
    testassert(class_getProperty(subcls, "superProp") != class_getProperty(submeta, "superProp"));
    testassert(class_getProperty(subcls, "subProtoProp") != class_getProperty(submeta, "subProtoProp"));
    testassert(class_getProperty(subcls, "superProtoProp") != class_getProperty(submeta, "superProtoProp"));

    // protocol properties

    prop = protocol_getProperty(@protocol(SubProto), "subProtoProp", YES, YES);
    testassert(prop);

    prop = protocol_getProperty(@protocol(SuperProto), "superProtoProp", YES, YES);
    testassert(prop == protocol_getProperty(@protocol(SubProto), "superProtoProp", YES, YES));

    prop = protocol_getProperty(@protocol(SuperProto), "subProtoProp", YES, YES);
    testassert(!prop);

    // protocol properties must not appear on the wrong side
    testassert(nil == protocol_getProperty(@protocol(SubProto), "subClassOnlyProtoProp", YES, YES));
    testassert(nil == protocol_getProperty(@protocol(SubProto), "subInstanceOnlyProtoProp", YES, NO));

    // protocol properties with the same name on both sides are distinct
    testassert(protocol_getProperty(@protocol(SubProto), "subProtoProp", YES, YES) != protocol_getProperty(@protocol(SubProto), "subProtoProp", YES, NO));
    testassert(protocol_getProperty(@protocol(SubProto), "superProtoProp", YES, YES) != protocol_getProperty(@protocol(SubProto), "superProtoProp", YES, NO));

    testassert(nil == protocol_getProperty(nil, "foo", YES, YES));
    testassert(nil == protocol_getProperty(@protocol(SuperProto), nil, YES, YES));
    testassert(nil == protocol_getProperty(nil, nil, YES, YES));
    testassert(nil == protocol_getProperty(@protocol(SuperProto), "superProtoProp", NO, YES));
    testassert(nil == protocol_getProperty(@protocol(SuperProto), "superProtoProp", NO, NO));

    succeed(__FILE__);
    return 0;
}
