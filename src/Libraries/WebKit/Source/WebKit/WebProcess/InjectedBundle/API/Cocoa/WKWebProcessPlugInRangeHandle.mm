/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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
#import "WKWebProcessPlugInRangeHandleInternal.h"

#import "InjectedBundleNodeHandle.h"
#import "WKDataDetectorTypesInternal.h"
#import "WKWebProcessPlugInFrameInternal.h"
#import <WebCore/DataDetection.h>
#import <WebCore/Range.h>
#import <WebCore/WebCoreObjCExtras.h>

@implementation WKWebProcessPlugInRangeHandle {
    API::ObjectStorage<WebKit::InjectedBundleRangeHandle> _rangeHandle;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKWebProcessPlugInRangeHandle.class, self))
        return;
    _rangeHandle->~InjectedBundleRangeHandle();
    [super dealloc];
}

+ (WKWebProcessPlugInRangeHandle *)rangeHandleWithJSValue:(JSValue *)value inContext:(JSContext *)context
{
    JSContextRef contextRef = [context JSGlobalContextRef];
    JSObjectRef objectRef = JSValueToObject(contextRef, [value JSValueRef], nullptr);
    return wrapper(WebKit::InjectedBundleRangeHandle::getOrCreate(contextRef, objectRef)).autorelease();
}

- (WKWebProcessPlugInFrame *)frame
{
    return wrapper(_rangeHandle->document()->documentFrame()).autorelease();
}

- (NSString *)text
{
    return _rangeHandle->text();
}

#if TARGET_OS_IPHONE

- (NSArray *)detectDataWithTypes:(WKDataDetectorTypes)types context:(NSDictionary *)context
{
#if ENABLE(DATA_DETECTION)
    return WebCore::DataDetection::detectContentInRange(makeSimpleRange(_rangeHandle->coreRange()), fromWKDataDetectorTypes(types), WebCore::DataDetection::extractReferenceDate(context));
#else
    return nil;
#endif
}

#endif

- (WebKit::InjectedBundleRangeHandle&)_rangeHandle
{
    return *_rangeHandle;
}

// MARK: WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_rangeHandle;
}

@end
