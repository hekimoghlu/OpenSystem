/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#import "config.h"
#import "ObjCRuntimeExtras.h"

#import <objc/runtime.h>
#import <wtf/StdLibExtras.h>

namespace WTF {

MallocSpan<Method, SystemMalloc> class_copyMethodListSpan(Class cls)
{
    unsigned methodCount = 0;
    auto* methods = class_copyMethodList(cls, &methodCount);
    return adoptMallocSpan<Method, SystemMalloc>(unsafeMakeSpan(methods, methodCount));
}

MallocSpan<__unsafe_unretained Protocol *, SystemMalloc> class_copyProtocolListSpan(Class cls)
{
    unsigned protocolCount = 0;
    auto* protocols = class_copyProtocolList(cls, &protocolCount);
    return adoptMallocSpan<__unsafe_unretained Protocol *, SystemMalloc>(unsafeMakeSpan(protocols, protocolCount));
}

MallocSpan<objc_property_t, SystemMalloc> class_copyPropertyListSpan(Class cls)
{
    unsigned propertyCount = 0;
    auto* properties = class_copyPropertyList(cls, &propertyCount);
    return adoptMallocSpan<objc_property_t, SystemMalloc>(unsafeMakeSpan(properties, propertyCount));
}

MallocSpan<Ivar, SystemMalloc> class_copyIvarListSpan(Class cls)
{
    unsigned ivarCount = 0;
    auto* ivars = class_copyIvarList(cls, &ivarCount);
    return adoptMallocSpan<Ivar, SystemMalloc>(unsafeMakeSpan(ivars, ivarCount));
}

MallocSpan<objc_method_description, SystemMalloc> protocol_copyMethodDescriptionListSpan(Protocol *protocol, BOOL isRequiredMethod, BOOL isInstanceMethod)
{
    unsigned methodCount = 0;
    auto* methods = protocol_copyMethodDescriptionList(protocol, isRequiredMethod, isInstanceMethod, &methodCount);
    return adoptMallocSpan<objc_method_description, SystemMalloc>(unsafeMakeSpan(methods, methodCount));
}

MallocSpan<objc_property_t, SystemMalloc> protocol_copyPropertyListSpan(Protocol *protocol)
{
    unsigned propertyCount = 0;
    auto* properties = protocol_copyPropertyList(protocol, &propertyCount);
    return adoptMallocSpan<objc_property_t, SystemMalloc>(unsafeMakeSpan(properties, propertyCount));
}

MallocSpan<__unsafe_unretained Protocol *, SystemMalloc> protocol_copyProtocolListSpan(Protocol *protocol)
{
    unsigned protocolCount = 0;
    auto* protocols = protocol_copyProtocolList(protocol, &protocolCount);
    return adoptMallocSpan<__unsafe_unretained Protocol *, SystemMalloc>(unsafeMakeSpan(protocols, protocolCount));
}

} // namespace WTF
