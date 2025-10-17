/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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
#include "config.h"
#include "APISerializedScriptValue.h"

#include "WKMutableArray.h"
#include "WKMutableDictionary.h"
#include "WKNumber.h"
#include "WKString.h"
#include <JavaScriptCore/JSRemoteInspector.h>
#include <JavaScriptCore/JSRetainPtr.h>

namespace API {

static constexpr auto SharedJSContextWKMaxIdleTime = 10_s;

class SharedJSContextWK {
public:
    static SharedJSContextWK& singleton()
    {
        static MainThreadNeverDestroyed<SharedJSContextWK> sharedContext;
        return sharedContext.get();
    }

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    JSRetainPtr<JSGlobalContextRef> ensureContext()
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
            m_context = adopt(JSGlobalContextCreate(nullptr));
            JSRemoteInspectorSetInspectionEnabledByDefault(previous);
            ALLOW_DEPRECATED_DECLARATIONS_END

            JSRemoteInspectorSetInspectionFollowsInternalPolicies(inspectionPreviouslyFollowedInternalPolicies);

            m_timer.startOneShot(SharedJSContextWKMaxIdleTime);
        }
        return m_context;
    }

    void releaseContextIfNecessary()
    {
        auto idleTime = MonotonicTime::now() - m_lastUseTime;
        if (idleTime < SharedJSContextWKMaxIdleTime) {
            // We lazily restart the timer if needed every 10 seconds instead of doing so every time ensureContext()
            // is called, for performance reasons.
            m_timer.startOneShot(SharedJSContextWKMaxIdleTime - idleTime);
            return;
        }
        m_context.clear();
    }

private:
    friend class NeverDestroyed<SharedJSContextWK, MainThreadAccessTraits>;

    SharedJSContextWK()
        : m_timer(RunLoop::main(), this, &SharedJSContextWK::releaseContextIfNecessary)
    {
    }

    JSRetainPtr<JSGlobalContextRef> m_context;
    RunLoop::Timer m_timer;
    MonotonicTime m_lastUseTime;
};

static WKRetainPtr<WKTypeRef> valueToWKObject(JSContextRef context, JSValueRef value)
{
    auto jsToWKString = [] (JSStringRef input) {
        size_t bufferSize = JSStringGetMaximumUTF8CStringSize(input);
        Vector<char> buffer(bufferSize);
        size_t utf8Length = JSStringGetUTF8CString(input, buffer.data(), bufferSize);
        ASSERT(buffer[utf8Length - 1] == '\0');
        return adoptWK(WKStringCreateWithUTF8CStringWithLength(buffer.data(), utf8Length - 1));
    };

    if (!JSValueIsObject(context, value)) {
        if (JSValueIsBoolean(context, value))
            return adoptWK(WKBooleanCreate(JSValueToBoolean(context, value)));
        if (JSValueIsNumber(context, value))
            return adoptWK(WKDoubleCreate(JSValueToNumber(context, value, nullptr)));
        if (JSValueIsString(context, value)) {
            JSStringRef jsString = JSValueToStringCopy(context, value, nullptr);
            WKRetainPtr result = jsToWKString(jsString);
            JSStringRelease(jsString);
            return result;
        }
        return nullptr;
    }

    JSObjectRef object = JSValueToObject(context, value, nullptr);

    if (JSValueIsArray(context, value)) {
        JSStringRef jsString = JSStringCreateWithUTF8CString("length");
        JSValueRef lengthPropertyName = JSValueMakeString(context, jsString);
        JSStringRelease(jsString);
        JSValueRef lengthValue = JSObjectGetPropertyForKey(context, object, lengthPropertyName, nullptr);
        double lengthDouble = JSValueToNumber(context, lengthValue, nullptr);
        if (lengthDouble < 0 || lengthDouble > static_cast<double>(std::numeric_limits<size_t>::max()))
            return nullptr;
        size_t length = lengthDouble;
        WKRetainPtr result = adoptWK(WKMutableArrayCreateWithCapacity(length));
        for (size_t i = 0; i < length; ++i)
            WKArrayAppendItem(result.get(), valueToWKObject(context, JSObjectGetPropertyAtIndex(context, object, i, nullptr)).get());
        return result;
    }

    JSPropertyNameArrayRef names = JSObjectCopyPropertyNames(context, object);
    size_t length = JSPropertyNameArrayGetCount(names);
    auto result = adoptWK(WKMutableDictionaryCreateWithCapacity(length));
    for (size_t i = 0; i < length; i++) {
        JSStringRef jsKey = JSPropertyNameArrayGetNameAtIndex(names, i);
        WKRetainPtr key = jsToWKString(jsKey);
        WKRetainPtr value = valueToWKObject(context, JSObjectGetPropertyForKey(context, object, JSValueMakeString(context, jsKey), nullptr));
        WKDictionarySetItem(result.get(), key.get(), value.get());
    }
    JSPropertyNameArrayRelease(names);

    return result;
}

WKRetainPtr<WKTypeRef> SerializedScriptValue::deserializeWK(WebCore::SerializedScriptValue& serializedScriptValue)
{
    ASSERT(RunLoop::isMain());
    JSRetainPtr context = SharedJSContextWK::singleton().ensureContext();
    ASSERT(context);

    JSValueRef value = serializedScriptValue.deserialize(context.get(), nullptr);
    if (!value)
        return nullptr;

    return valueToWKObject(context.get(), value);
}

Vector<uint8_t> SerializedScriptValue::serializeCryptoKey(const WebCore::CryptoKey& key)
{
    ASSERT(RunLoop::isMain());
    JSRetainPtr context = SharedJSContextWK::singleton().ensureContext();
    ASSERT(context);

    return WebCore::SerializedScriptValue::serializeCryptoKey(context.get(), key);
}

} // API
