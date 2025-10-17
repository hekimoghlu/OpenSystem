/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 19, 2024.
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

#include <JavaScriptCore/APICast.h>
#include <JavaScriptCore/JSBase.h>
#include <JavaScriptCore/JSContextPrivate.h>
#include <JavaScriptCore/JSGlobalObjectInlines.h>
#include <JavaScriptCore/JSRemoteInspector.h>
#include <jsc/JSCContextPrivate.h>
#include <jsc/JSCValuePrivate.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RunLoop.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/glib/RunLoopSourcePriority.h>

namespace API {

static constexpr auto sharedJSContextMaxIdleTime = 10_s;

class SharedJSContext {
public:
    static SharedJSContext& singleton()
    {
        static NeverDestroyed<SharedJSContext> sharedContext;
        return sharedContext.get();
    }

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    JSCContext* ensureContext()
    {
        m_lastUseTime = MonotonicTime::now();
        if (!m_context) {
            bool previous = JSRemoteInspectorGetInspectionEnabledByDefault();
            JSRemoteInspectorSetInspectionEnabledByDefault(false);
            m_context = adoptGRef(jsc_context_new());
            JSRemoteInspectorSetInspectionEnabledByDefault(previous);

            m_timer.startOneShot(sharedJSContextMaxIdleTime);
        }
        return m_context.get();
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
    friend class NeverDestroyed<SharedJSContext>;

    SharedJSContext()
        : m_timer(RunLoop::main(), this, &SharedJSContext::releaseContextIfNecessary)
    {
        m_timer.setPriority(RunLoopSourcePriority::ReleaseUnusedResourcesTimer);
    }

    GRefPtr<JSCContext> m_context;
    RunLoop::Timer m_timer;
    MonotonicTime m_lastUseTime;
};

JSCContext* SerializedScriptValue::sharedJSCContext()
{
    return SharedJSContext::singleton().ensureContext();
}

GRefPtr<JSCValue> SerializedScriptValue::deserialize(WebCore::SerializedScriptValue& serializedScriptValue)
{
    ASSERT(RunLoop::isMain());

    auto* context = sharedJSCContext();
    return jscContextGetOrCreateValue(context, serializedScriptValue.deserialize(jscContextGetJSContext(context), nullptr));
}

static GRefPtr<JSCValue> valueFromGVariant(JSCContext* context, GVariant* variant)
{
    if (g_variant_is_of_type(variant, G_VARIANT_TYPE("a{sv}"))) {
        auto result = adoptGRef(jsc_value_new_object(context, nullptr, nullptr));
        GVariantIter iter;
        g_variant_iter_init(&iter, variant);
        const char* key;
        GVariant* value;
        while (g_variant_iter_loop(&iter, "{&sv}", &key, &value)) {
            if (!key)
                continue;
            auto jsValue = valueFromGVariant(context, value);
            if (jsValue)
                jsc_value_object_set_property(result.get(), key, jsValue.get());
        }
        return result;
    }

    if (g_variant_is_of_type(variant, G_VARIANT_TYPE_UINT32))
        return adoptGRef(jsc_value_new_number(context, g_variant_get_uint32(variant)));
    if (g_variant_is_of_type(variant, G_VARIANT_TYPE_INT32))
        return adoptGRef(jsc_value_new_number(context, g_variant_get_int32(variant)));
    if (g_variant_is_of_type(variant, G_VARIANT_TYPE_UINT64))
        return adoptGRef(jsc_value_new_number(context, g_variant_get_uint64(variant)));
    if (g_variant_is_of_type(variant, G_VARIANT_TYPE_INT64))
        return adoptGRef(jsc_value_new_number(context, g_variant_get_int64(variant)));
    if (g_variant_is_of_type(variant, G_VARIANT_TYPE_INT16))
        return adoptGRef(jsc_value_new_number(context, g_variant_get_int16(variant)));
    if (g_variant_is_of_type(variant, G_VARIANT_TYPE_UINT16))
        return adoptGRef(jsc_value_new_number(context, g_variant_get_uint16(variant)));
    if (g_variant_is_of_type(variant, G_VARIANT_TYPE_DOUBLE))
        return adoptGRef(jsc_value_new_number(context, g_variant_get_double(variant)));

    if (g_variant_is_of_type(variant, G_VARIANT_TYPE_STRING))
        return adoptGRef(jsc_value_new_string(context, g_variant_get_string(variant, nullptr)));

    return nullptr;
}

static RefPtr<WebCore::SerializedScriptValue> coreValueFromGVariant(GVariant* variant)
{
    if (!variant)
        return nullptr;

    ASSERT(RunLoop::isMain());
    auto* context = SerializedScriptValue::sharedJSCContext();
    auto value = valueFromGVariant(context, variant);
    if (!value)
        return nullptr;

    auto globalObject = toJS(jscContextGetJSContext(context));
    ASSERT(globalObject);
    JSC::JSLockHolder lock(globalObject);

    return WebCore::SerializedScriptValue::create(*globalObject, toJS(globalObject, jscValueGetJSValue(value.get())));
}

RefPtr<SerializedScriptValue> SerializedScriptValue::createFromGVariant(GVariant* object)
{
    auto coreValue = coreValueFromGVariant(object);
    if (!coreValue)
        return nullptr;
    return create(coreValue.releaseNonNull());
}

RefPtr<SerializedScriptValue> SerializedScriptValue::createFromJSCValue(JSCValue* value)
{
    ASSERT(jsc_value_get_context(value) == sharedJSCContext());
    return create(jscContextGetJSContext(jsc_value_get_context(value)), jscValueGetJSValue(value), nullptr);
}

}; // namespace API
