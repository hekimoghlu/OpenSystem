/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 12, 2022.
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

#include "Event.h"
#include <wtf/URL.h>

namespace WebCore {

class DOMPromise;
class Document;

class MerchantValidationEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MerchantValidationEvent);
public:
    struct Init final : EventInit {
        String methodName;
        String validationURL;
    };

    static Ref<MerchantValidationEvent> create(const AtomString& type, const String& methodName, URL&& validationURL);
    static ExceptionOr<Ref<MerchantValidationEvent>> create(Document&, const AtomString& type, Init&&);

    const String& methodName() const { return m_methodName; }
    const String& validationURL() const { return m_validationURL.string(); }
    ExceptionOr<void> complete(Ref<DOMPromise>&&);

private:
    MerchantValidationEvent(const AtomString& type, const String& methodName, URL&& validationURL);
    MerchantValidationEvent(const AtomString& type, String&& methodName, URL&& validationURL, Init&&);

    bool m_isCompleted { false };
    String m_methodName;
    URL m_validationURL;
};

} // namespace WebCore

#endif // ENABLE(PAYMENT_REQUEST)
