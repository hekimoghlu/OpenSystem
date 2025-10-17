/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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

#if ENABLE(PAYMENT_REQUEST)

#include "JSValueInWrappedObject.h"
#include "PaymentRequestUpdateEvent.h"
#include <JavaScriptCore/Strong.h>
#include <variant>
#include <wtf/text/WTFString.h>

namespace JSC {
class JSObject;
}

namespace WebCore {

class PaymentMethodChangeEvent final : public PaymentRequestUpdateEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PaymentMethodChangeEvent);
public:
    template<typename... Args> static Ref<PaymentMethodChangeEvent> create(Args&&... args)
    {
        return adoptRef(*new PaymentMethodChangeEvent(std::forward<Args>(args)...));
    }

    using MethodDetailsFunction = std::function<JSC::Strong<JSC::JSObject>(JSC::JSGlobalObject&)>;
    using MethodDetailsType = std::variant<JSValueInWrappedObject, MethodDetailsFunction>;

    const String& methodName() const { return m_methodName; }
    const MethodDetailsType& methodDetails() const { return m_methodDetails; }
    JSValueInWrappedObject& cachedMethodDetails() { return m_cachedMethodDetails; }

    struct Init final : PaymentRequestUpdateEventInit {
        String methodName;
        JSC::Strong<JSC::JSObject> methodDetails;
    };

private:
    PaymentMethodChangeEvent(const AtomString& type, Init&&);
    PaymentMethodChangeEvent(const AtomString& type, const String& methodName, MethodDetailsFunction&&);

    String m_methodName;
    MethodDetailsType m_methodDetails;
    JSValueInWrappedObject m_cachedMethodDetails;
};

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
