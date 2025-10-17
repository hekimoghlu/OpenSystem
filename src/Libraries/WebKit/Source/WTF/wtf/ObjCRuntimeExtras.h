/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
#pragma once

#import <objc/message.h>
#import <wtf/MallocSpan.h>
#import <wtf/SystemMalloc.h>

#ifdef __cplusplus

template<typename ReturnType, typename... ArgumentTypes>
ReturnType wtfObjCMsgSend(id target, SEL selector, ArgumentTypes... arguments)
{
    return reinterpret_cast<ReturnType (*)(id, SEL, ArgumentTypes...)>(objc_msgSend)(target, selector, arguments...);
}

template<typename ReturnType, typename... ArgumentTypes>
ReturnType wtfCallIMP(IMP implementation, id target, SEL selector, ArgumentTypes... arguments)
{
    return reinterpret_cast<ReturnType (*)(id, SEL, ArgumentTypes...)>(implementation)(target, selector, arguments...);
}

namespace WTF {

WTF_EXPORT_PRIVATE MallocSpan<Method, SystemMalloc> class_copyMethodListSpan(Class);
WTF_EXPORT_PRIVATE MallocSpan<__unsafe_unretained Protocol *, SystemMalloc> class_copyProtocolListSpan(Class);
WTF_EXPORT_PRIVATE MallocSpan<objc_property_t, SystemMalloc> class_copyPropertyListSpan(Class);
WTF_EXPORT_PRIVATE MallocSpan<Ivar, SystemMalloc> class_copyIvarListSpan(Class);
WTF_EXPORT_PRIVATE MallocSpan<objc_method_description, SystemMalloc> protocol_copyMethodDescriptionListSpan(Protocol *, BOOL isRequiredMethod, BOOL isInstanceMethod);
WTF_EXPORT_PRIVATE MallocSpan<objc_property_t, SystemMalloc> protocol_copyPropertyListSpan(Protocol *);
WTF_EXPORT_PRIVATE MallocSpan<__unsafe_unretained Protocol *, SystemMalloc> protocol_copyProtocolListSpan(Protocol *);

} // namespace WTF

using WTF::class_copyIvarListSpan;
using WTF::class_copyMethodListSpan;
using WTF::class_copyPropertyListSpan;
using WTF::class_copyProtocolListSpan;
using WTF::protocol_copyMethodDescriptionListSpan;
using WTF::protocol_copyPropertyListSpan;
using WTF::protocol_copyProtocolListSpan;

#endif // __cplusplus
