/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
#import "APISerializedScriptValue.h"

#import <JavaScriptCore/APICast.h>
#import <JavaScriptCore/JSContextPrivate.h>
#import <JavaScriptCore/JSGlobalObjectInlines.h>
#import <JavaScriptCore/JSRemoteInspector.h>
#import <JavaScriptCore/JSRetainPtr.h>
#import <JavaScriptCore/JSValue.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RunLoop.h>

namespace API {

static constexpr auto sharedJSContextMaxIdleTime = 10_s;

class SharedJSContext {
public:
    static SharedJSContext& singleton()
    {
        static MainThreadNeverDestroyed<SharedJSContext> sharedContext;
        return sharedContext.get();
    }

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    RetainPtr<JSContext> ensureContext()
    {
        m_lastUseTime = MonotonicTime::now();
        if (!m_context) {
            bool inspectionPreviouslyFollowedInternalPolicies = JSRemoteInspectorGetInspectionFollowsInternalPolicies();
            JSRemoteInspectorSetInspectionFollowsInternalPolicies(false);

            // FIXME: rdar://100738357 Remote Web Inspector: Remove use of JSRemoteInspectorGetInspectionEnabledByDefault
            // and JSRemoteInspectorSetInspectionEnabledByDefault once the default state is always false.
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
            bool previous = JSRemoteInspectorGetInspectionEnabledByDefault();
            JSRemoteInspectorSetInspectionEnabledByDefault(false);
            m_context = adoptNS([[JSContext alloc] init]);
            JSRemoteInspectorSetInspectionEnabledByDefault(previous);
ALLOW_DEPRECATED_DECLARATIONS_END

            JSRemoteInspectorSetInspectionFollowsInternalPolicies(inspectionPreviouslyFollowedInternalPolicies);

            m_timer.startOneShot(sharedJSContextMaxIdleTime);
        }
        return m_context;
    }

    void releaseContextIfNecessary()
    {
        auto idleTime = MonotonicTime::now() - m_lastUseTime;
        if (idleTime < sharedJSContextMaxIdleTime) {
            // We lazily restart the timer if needed every 10 seconds instead of doing so every time ensureContext()
            // is called, for performance reasons.
            m_timer.startOneShot(sharedJSContextMaxIdleTime - idleTime);
            return;
        }
        m_context.clear();
    }

private:
    friend class NeverDestroyed<SharedJSContext, MainThreadAccessTraits>;

    SharedJSContext()
        : m_timer(RunLoop::main(), this, &SharedJSContext::releaseContextIfNecessary)
    {
    }

    RetainPtr<JSContext> m_context;
    RunLoop::Timer m_timer;
    MonotonicTime m_lastUseTime;
};

id SerializedScriptValue::deserialize(WebCore::SerializedScriptValue& serializedScriptValue)
{
    ASSERT(RunLoop::isMain());
    RetainPtr context = SharedJSContext::singleton().ensureContext();
    JSRetainPtr globalContextRef = [context JSGlobalContextRef];

    JSValueRef valueRef = serializedScriptValue.deserialize(globalContextRef.get(), nullptr);
    if (!valueRef)
        return nil;

    JSValue *value = [JSValue valueWithJSValueRef:valueRef inContext:context.get()];
    return value.toObject;
}

static bool validateObject(id argument)
{
    if ([argument isKindOfClass:[NSString class]] || [argument isKindOfClass:[NSNumber class]] || [argument isKindOfClass:[NSDate class]] || [argument isKindOfClass:[NSNull class]])
        return true;

    if ([argument isKindOfClass:[NSArray class]]) {
        __block BOOL valid = true;

        [argument enumerateObjectsUsingBlock:^(id object, NSUInteger, BOOL *stop) {
            if (!validateObject(object)) {
                valid = false;
                *stop = YES;
            }
        }];

        return valid;
    }

    if ([argument isKindOfClass:[NSDictionary class]]) {
        __block bool valid = true;

        [argument enumerateKeysAndObjectsUsingBlock:^(id key, id value, BOOL *stop) {
            if (!validateObject(key) || !validateObject(value)) {
                valid = false;
                *stop = YES;
            }
        }];

        return valid;
    }

    return false;
}

static RefPtr<WebCore::SerializedScriptValue> coreValueFromNSObject(id object)
{
    if (object && !validateObject(object))
        return nullptr;

    ASSERT(RunLoop::isMain());
    RetainPtr context = SharedJSContext::singleton().ensureContext();
    JSValue *value = [JSValue valueWithObject:object inContext:context.get()];
    if (!value)
        return nullptr;

    JSRetainPtr globalContextRef = [context JSGlobalContextRef];
    auto globalObject = toJS(globalContextRef.get());
    ASSERT(globalObject);
    JSC::JSLockHolder lock(globalObject);

    return WebCore::SerializedScriptValue::create(*globalObject, toJS(globalObject, [value JSValueRef]));
}

RefPtr<SerializedScriptValue> SerializedScriptValue::createFromNSObject(id object)
{
    auto coreValue = coreValueFromNSObject(object);
    if (!coreValue)
        return nullptr;
    return create(coreValue.releaseNonNull());
}

} // API
