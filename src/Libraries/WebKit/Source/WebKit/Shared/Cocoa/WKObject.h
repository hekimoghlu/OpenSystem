/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#import "Logging.h"
#import "WKFoundation.h"

#import <type_traits>
#import <wtf/ObjCRuntimeExtras.h>
#import <wtf/RefPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/spi/cocoa/objcSPI.h>

namespace API {

class Object;

template<typename ObjectClass> struct ObjectStorage {
    ObjectClass* get() { return reinterpret_cast<ObjectClass*>(&data); }
    ObjectClass& operator*() { return *get(); }
    ObjectClass* operator->() { return get(); }

    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    typename std::aligned_storage<sizeof(ObjectClass), std::alignment_of<ObjectClass>::value>::type data;
    ALLOW_DEPRECATED_DECLARATIONS_END
};

API::Object* unwrap(void*);
void* wrap(API::Object*);

}

namespace WebKit {

template<typename WrappedObjectClass> struct WrapperTraits;

template<typename DestinationClass, typename SourceClass> inline DestinationClass *checkedObjCCast(SourceClass *source)
{
    return checked_objc_cast<DestinationClass>(source);
}

template<typename ObjectClass> inline typename WrapperTraits<ObjectClass>::WrapperClass *wrapper(ObjectClass& object)
{
    return checkedObjCCast<typename WrapperTraits<ObjectClass>::WrapperClass>(object.wrapper());
}

template<typename ObjectClass> inline typename WrapperTraits<ObjectClass>::WrapperClass *wrapper(ObjectClass* object)
{
    return object ? wrapper(*object) : nil;
}

template<typename ObjectClass> inline typename WrapperTraits<ObjectClass>::WrapperClass *wrapper(const Ref<ObjectClass>& object)
{
    return wrapper(object.get());
}

template<typename ObjectClass> inline typename WrapperTraits<ObjectClass>::WrapperClass *wrapper(const RefPtr<ObjectClass>& object)
{
    return wrapper(object.get());
}

template<typename ObjectClass> inline RetainPtr<typename WrapperTraits<ObjectClass>::WrapperClass> wrapper(Ref<ObjectClass>&& object)
{
    return wrapper(object.get());
}

template<typename ObjectClass> inline RetainPtr<typename WrapperTraits<ObjectClass>::WrapperClass> wrapper(RefPtr<ObjectClass>&& object)
{
    return object ? wrapper(object.releaseNonNull()) : nil;
}

}

namespace API {

using WebKit::wrapper;

}

@protocol WKObject <NSObject>

@property (readonly) API::Object& _apiObject;

@end

@interface WKObject : NSProxy <WKObject>

- (NSObject *)_web_createTarget NS_RETURNS_RETAINED;

@end

#if HAVE(OBJC_CUSTOM_DEALLOC)

// This macro ensures WebKit ObjC objects of a specified class are deallocated on the main thread.
// Use this macro in the ObjC implementation file.

#define WK_OBJECT_DEALLOC_ON_MAIN_THREAD(objcClass) \
+ (void)initialize \
{ \
    if (self == objcClass.class) \
        _class_setCustomDeallocInitiation(self); \
} \
\
- (void)_objc_initiateDealloc \
{ \
    if (isMainRunLoop()) \
        _objc_deallocOnMainThreadHelper((__bridge void *)self); \
    else \
        dispatch_async_f(dispatch_get_main_queue(), (__bridge void *)self, _objc_deallocOnMainThreadHelper); \
} \
\
using __thisIsHereToForceASemicolonAfterThisMacro UNUSED_TYPE_ALIAS = int

// This macro ensures WebKit ObjC objects and their C++ implementation are safely deallocated on the main thread.
// Use this macro in the ObjC implementation file if you don't require a custom dealloc method.

#define WK_OBJECT_DEALLOC_IMPL_ON_MAIN_THREAD(objcClass, implClass, storageVar) \
WK_OBJECT_DEALLOC_ON_MAIN_THREAD(objcClass); \
\
- (void)dealloc \
{ \
    ASSERT(isMainRunLoop()); \
    SUPPRESS_UNCOUNTED_ARG storageVar->~implClass(); \
} \
\
using __thisIsHereToForceASemicolonAfterThisMacro UNUSED_TYPE_ALIAS = int

#else

#define WK_OBJECT_DEALLOC_ON_MAIN_THREAD(objcClass) \
using __thisIsHereToForceASemicolonAfterThisMacro UNUSED_TYPE_ALIAS = int

#define WK_OBJECT_DEALLOC_IMPL_ON_MAIN_THREAD(objcClass, implClass, storageVar) \
- (void)dealloc \
{ \
    ASSERT(isMainRunLoop()); \
    storageVar->~implClass(); \
} \
\
using __thisIsHereToForceASemicolonAfterThisMacro UNUSED_TYPE_ALIAS = int

#endif // HAVE(OBJC_CUSTOM_DEALLOC)

#define WK_OBJECT_DISABLE_DISABLE_KVC_IVAR_ACCESS \
+ (BOOL)accessInstanceVariablesDirectly \
{ \
    return !linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::ThrowOnKVCInstanceVariableAccess); \
} \
using __thisIsHereToForceASemicolonAfterThisMacro UNUSED_TYPE_ALIAS = int
