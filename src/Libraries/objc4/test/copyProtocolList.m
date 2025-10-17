/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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

#include "test.h"
#include <string.h>
#include <malloc/malloc.h>
#include <objc/objc-runtime.h>

@protocol Proto1 
+(id)proto1ClassMethod;
-(id)proto1InstanceMethod;
@end

void noNullEntries(Protocol * _Nonnull __unsafe_unretained * _Nullable protolist,
                   unsigned int count)
{
    for (unsigned int i = 0; i != count; ++i) {
        testassert(protolist[i]);
        testassert(protocol_getName(protolist[i]));
        testprintf("Protocol[%d/%d]: %p %s\n", i, count, protolist[i], protocol_getName(protolist[i]));
    }
}

Protocol* getProtocol(Protocol * _Nonnull __unsafe_unretained * _Nullable protolist,
                      unsigned int count, const char* name) {
    for (unsigned int i = 0; i != count; ++i) {
        if (!strcmp(protocol_getName(protolist[i]), name))
            return protolist[i];
    }
    return nil;
}

int main()
{
    Protocol * _Nonnull __unsafe_unretained * _Nullable protolist;
    unsigned int count;

    count = 100;
    protolist = objc_copyProtocolList(&count);
    testassert(protolist);
    testassert(count != 0);
    testassert(malloc_size(protolist) >= (count * sizeof(Protocol*)));
    noNullEntries(protolist, count);
    testassert(protolist[count] == nil);
    // Check for a shared cache protocol, ie, the one we know comes from libobjc
    testassert(getProtocol(protolist, count, "NSObject"));
    // Test for a protocol we know isn't in the cache
    testassert(getProtocol(protolist, count, "Proto1") == @protocol(Proto1));
    // Test for a protocol we know isn't there
    testassert(!getProtocol(protolist, count, "Proto2"));
    free(protolist);

    // Now add it
    Protocol* newproto = objc_allocateProtocol("Proto2");
    objc_registerProtocol(newproto);

    Protocol * _Nonnull __unsafe_unretained * _Nullable newProtolist;
    unsigned int newCount;

    newCount = 100;
    newProtolist = objc_copyProtocolList(&newCount);
    testassert(newProtolist);
    testassert(newCount == (count + 1));
    testassert(getProtocol(newProtolist, newCount, "Proto2"));
    free(newProtolist);
    

    succeed(__FILE__);
    return 0;
}
