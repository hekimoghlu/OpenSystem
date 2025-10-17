/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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

#include "APIObject.h"
#include "WKRetainPtr.h"
#include <WebCore/CryptoKey.h>
#include <WebCore/SerializedScriptValue.h>
#include <wtf/RefPtr.h>

#if USE(GLIB)
#include <wtf/glib/GRefPtr.h>

typedef struct _GVariant GVariant;
typedef struct _JSCContext JSCContext;
typedef struct _JSCValue JSCValue;
#endif

typedef const void* WKTypeRef;

namespace API {

class SerializedScriptValue : public API::ObjectImpl<API::Object::Type::SerializedScriptValue> {
public:
    static Ref<SerializedScriptValue> create(Ref<WebCore::SerializedScriptValue>&& serializedValue)
    {
        return adoptRef(*new SerializedScriptValue(WTFMove(serializedValue)));
    }
    
    static RefPtr<SerializedScriptValue> create(JSContextRef context, JSValueRef value, JSValueRef* exception)
    {
        RefPtr<WebCore::SerializedScriptValue> serializedValue = WebCore::SerializedScriptValue::create(context, value, exception);
        if (!serializedValue)
            return nullptr;
        return adoptRef(*new SerializedScriptValue(serializedValue.releaseNonNull()));
    }
    
    static Ref<SerializedScriptValue> createFromWireBytes(std::span<const uint8_t> buffer)
    {
        return adoptRef(*new SerializedScriptValue(WebCore::SerializedScriptValue::createFromWireBytes(Vector<uint8_t>(buffer))));
    }
    
    JSValueRef deserialize(JSContextRef context, JSValueRef* exception)
    {
        return m_serializedScriptValue->deserialize(context, exception);
    }

    static WKRetainPtr<WKTypeRef> deserializeWK(WebCore::SerializedScriptValue&);
    static Vector<uint8_t> serializeCryptoKey(const WebCore::CryptoKey&);

#if PLATFORM(COCOA) && defined(__OBJC__)
    static id deserialize(WebCore::SerializedScriptValue&);
    static RefPtr<SerializedScriptValue> createFromNSObject(id);
#endif

#if USE(GLIB)
    static JSCContext* sharedJSCContext();
    static GRefPtr<JSCValue> deserialize(WebCore::SerializedScriptValue&);
    static RefPtr<SerializedScriptValue> createFromGVariant(GVariant*);
    static RefPtr<SerializedScriptValue> createFromJSCValue(JSCValue*);
#endif

    std::span<const uint8_t> dataReference() const { return m_serializedScriptValue->wireBytes(); }

    WebCore::SerializedScriptValue& internalRepresentation() { return m_serializedScriptValue.get(); }

private:
    explicit SerializedScriptValue(Ref<WebCore::SerializedScriptValue>&& serializedScriptValue)
        : m_serializedScriptValue(WTFMove(serializedScriptValue))
    {
    }

    Ref<WebCore::SerializedScriptValue> m_serializedScriptValue;
};
    
}

SPECIALIZE_TYPE_TRAITS_API_OBJECT(SerializedScriptValue);
