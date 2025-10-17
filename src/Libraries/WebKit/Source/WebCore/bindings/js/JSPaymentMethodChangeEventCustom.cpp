/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#include "JSPaymentMethodChangeEvent.h"

#if ENABLE(PAYMENT_REQUEST)

namespace WebCore {

JSC::JSValue JSPaymentMethodChangeEvent::methodDetails(JSC::JSGlobalObject& lexicalGlobalObject) const
{
    auto throwScope = DECLARE_THROW_SCOPE(lexicalGlobalObject.vm());
    return cachedPropertyValue(throwScope, lexicalGlobalObject, *this, wrapped().cachedMethodDetails(), [this, &lexicalGlobalObject](JSC::ThrowScope&) {
        return WTF::switchOn(wrapped().methodDetails(), [](const JSValueInWrappedObject& methodDetails) -> JSC::JSValue {
            return methodDetails.getValue(JSC::jsNull());
        }, [&lexicalGlobalObject](const PaymentMethodChangeEvent::MethodDetailsFunction& function) -> JSC::JSValue {
            return function(lexicalGlobalObject).get();
        });
    });
}

template<typename Visitor>
void JSPaymentMethodChangeEvent::visitAdditionalChildren(Visitor& visitor)
{
    WTF::switchOn(wrapped().methodDetails(), [&visitor](const JSValueInWrappedObject& methodDetails) {
        methodDetails.visit(visitor);
    }, [](const PaymentMethodChangeEvent::MethodDetailsFunction&) {
    });

    wrapped().cachedMethodDetails().visit(visitor);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSPaymentMethodChangeEvent);

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
