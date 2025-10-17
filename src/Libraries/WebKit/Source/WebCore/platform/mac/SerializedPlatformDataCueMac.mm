/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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
#import "SerializedPlatformDataCueMac.h"

#if ENABLE(VIDEO) && ENABLE(DATACUE_VALUE)

#import "JSDOMConvertBufferSource.h"
#import <AVFoundation/AVMetadataItem.h>
#import <Foundation/NSString.h>
#import <JavaScriptCore/APICast.h>
#import <JavaScriptCore/ArrayBuffer.h>
#import <JavaScriptCore/JSArrayBuffer.h>
#import <JavaScriptCore/JSContextRef.h>
#import <JavaScriptCore/JSObjectRef.h>
#import <JavaScriptCore/JavaScriptCore.h>
#import <objc/runtime.h>
#import <wtf/cocoa/SpanCocoa.h>

#import <pal/cocoa/AVFoundationSoftLink.h>

namespace WebCore {

#if JSC_OBJC_API_ENABLED
static JSValue *jsValueWithDataInContext(NSData *, JSContext *);
static JSValue *jsValueWithArrayInContext(NSArray *, JSContext *);
static JSValue *jsValueWithDictionaryInContext(NSDictionary *, JSContext *);
static JSValue *jsValueWithAVMetadataItemInContext(AVMetadataItem *, JSContext *);
static JSValue *jsValueWithValueInContext(id, JSContext *);
#endif

Ref<SerializedPlatformDataCue> SerializedPlatformDataCue::create(SerializedPlatformDataCueValue&& value)
{
    return adoptRef(*new SerializedPlatformDataCueMac(WTFMove(value)));
}

SerializedPlatformDataCueMac::SerializedPlatformDataCueMac(SerializedPlatformDataCueValue&& value)
    : SerializedPlatformDataCue()
    , m_value(WTFMove(value))
{
}

RefPtr<ArrayBuffer> SerializedPlatformDataCueMac::data() const
{
    return nullptr;
}

JSC::JSValue SerializedPlatformDataCueMac::deserialize(JSC::JSGlobalObject* lexicalGlobalObject) const
{
#if JSC_OBJC_API_ENABLED
    auto dictionary = m_value.toNSDictionary();
    if (!dictionary)
        return JSC::jsNull();

    JSGlobalContextRef jsGlobalContextRef = toGlobalRef(lexicalGlobalObject);
    JSContext *jsContext = [JSContext contextWithJSGlobalContextRef:jsGlobalContextRef];
    JSValue *serializedValue = jsValueWithValueInContext(dictionary.get(), jsContext);

    return toJS(lexicalGlobalObject, [serializedValue JSValueRef]);
#else
    UNUSED_PARAM(lexicalGlobalObject);
    return JSC::jsNull();
#endif
}

bool SerializedPlatformDataCueMac::isEqual(const SerializedPlatformDataCue& other) const
{
    return m_value == toSerializedPlatformDataCueMac(&other)->m_value;
}

SerializedPlatformDataCueMac* toSerializedPlatformDataCueMac(SerializedPlatformDataCue* rep)
{
    return const_cast<SerializedPlatformDataCueMac*>(toSerializedPlatformDataCueMac(const_cast<const SerializedPlatformDataCue*>(rep)));
}

const SerializedPlatformDataCueMac* toSerializedPlatformDataCueMac(const SerializedPlatformDataCue* rep)
{
    return static_cast<const SerializedPlatformDataCueMac*>(rep);
}

const UncheckedKeyHashSet<RetainPtr<Class>>& SerializedPlatformDataCueMac::allowedClassesForNativeValues()
{
    static NeverDestroyed<UncheckedKeyHashSet<RetainPtr<Class>>> allowedClasses(UncheckedKeyHashSet<RetainPtr<Class>> { [NSString class], [NSNumber class], [NSLocale class], [NSDictionary class], [NSArray class], [NSData class] });
    return allowedClasses;
}

SerializedPlatformDataCueValue SerializedPlatformDataCueMac::encodableValue() const
{
    return m_value;
}

#if JSC_OBJC_API_ENABLED
static JSValue *jsValueWithValueInContext(id value, JSContext *context)
{
    if ([value isKindOfClass:[NSString class]] || [value isKindOfClass:[NSNumber class]])
        return [JSValue valueWithObject:value inContext:context];

    if ([value isKindOfClass:[NSLocale class]])
        return [JSValue valueWithObject:[value localeIdentifier] inContext:context];

    if ([value isKindOfClass:[NSDictionary class]])
        return jsValueWithDictionaryInContext(value, context);

    if ([value isKindOfClass:[NSArray class]])
        return jsValueWithArrayInContext(value, context);

    if ([value isKindOfClass:[NSData class]])
        return jsValueWithDataInContext(value, context);

    if ([value isKindOfClass:PAL::getAVMetadataItemClass()])
        return jsValueWithAVMetadataItemInContext(value, context);

    return nil;
}

static JSValue *jsValueWithDataInContext(NSData *data, JSContext *context)
{
    auto dataArray = ArrayBuffer::tryCreate(span(data));

    auto* lexicalGlobalObject = toJS([context JSGlobalContextRef]);
    JSC::JSValue array = toJS(lexicalGlobalObject, JSC::jsCast<JSDOMGlobalObject*>(lexicalGlobalObject), dataArray.get());

    return [JSValue valueWithJSValueRef:toRef(lexicalGlobalObject, array) inContext:context];
}

static JSValue *jsValueWithArrayInContext(NSArray *array, JSContext *context)
{
    JSValueRef exception = 0;
    JSValue *result = [JSValue valueWithNewArrayInContext:context];
    JSObjectRef resultObject = JSValueToObject([context JSGlobalContextRef], [result JSValueRef], &exception);
    if (exception)
        return [JSValue valueWithUndefinedInContext:context];

    NSUInteger count = [array count];
    for (NSUInteger i = 0; i < count; ++i) {
        JSValue *value = jsValueWithValueInContext([array objectAtIndex:i], context);
        if (!value)
            continue;

        JSObjectSetPropertyAtIndex([context JSGlobalContextRef], resultObject, (unsigned)i, [value JSValueRef], &exception);
        if (exception)
            continue;
    }

    return result;
}

static JSValue *jsValueWithDictionaryInContext(NSDictionary *dictionary, JSContext *context)
{
    JSValueRef exception = 0;
    JSValue *result = [JSValue valueWithNewObjectInContext:context];
    JSObjectRef resultObject = JSValueToObject([context JSGlobalContextRef], [result JSValueRef], &exception);
    if (exception)
        return [JSValue valueWithUndefinedInContext:context];

    for (id key in [dictionary keyEnumerator]) {
        if (![key isKindOfClass:[NSString class]])
            continue;

        JSValue *value = jsValueWithValueInContext([dictionary objectForKey:key], context);
        if (!value)
            continue;

        auto name = OpaqueJSString::tryCreate(key);
        JSObjectSetProperty([context JSGlobalContextRef], resultObject, name.get(), [value JSValueRef], 0, &exception);
        if (exception)
            continue;
    }

    return result;
}

static JSValue *jsValueWithAVMetadataItemInContext(AVMetadataItem *item, JSContext *context)
{
    return jsValueWithDictionaryInContext(SerializedPlatformDataCueValue(item).toNSDictionary().get(), context);
}
#endif

} // namespace WebCore

#endif
