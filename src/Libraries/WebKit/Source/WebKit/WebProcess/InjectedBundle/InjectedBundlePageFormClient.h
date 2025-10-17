/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#ifndef InjectedBundlePageFormClient_h
#define InjectedBundlePageFormClient_h

#include "APIClient.h"
#include "APIInjectedBundleFormClient.h"
#include "WKBundlePageFormClient.h"
#include <wtf/TZoneMalloc.h>

namespace API {

template<> struct ClientTraits<WKBundlePageFormClientBase> {
    typedef std::tuple<WKBundlePageFormClientV0, WKBundlePageFormClientV1, WKBundlePageFormClientV2, WKBundlePageFormClientV3> Versions;
};
}

namespace WebKit {

class InjectedBundlePageFormClient : public API::Client<WKBundlePageFormClientBase>, public API::InjectedBundle::FormClient {
    WTF_MAKE_TZONE_ALLOCATED(InjectedBundlePageFormClient);
public:
    explicit InjectedBundlePageFormClient(const WKBundlePageFormClientBase*);

    void didFocusTextField(WebPage*, WebCore::HTMLInputElement&, WebFrame*) override;
    void textFieldDidBeginEditing(WebPage*, WebCore::HTMLInputElement&, WebFrame*) override;
    void textFieldDidEndEditing(WebPage*, WebCore::HTMLInputElement&, WebFrame*) override;
    void textDidChangeInTextField(WebPage*, WebCore::HTMLInputElement&, WebFrame*, bool initiatedByUserTyping) override;
    void textDidChangeInTextArea(WebPage*, WebCore::HTMLTextAreaElement&, WebFrame*) override;
    bool shouldPerformActionInTextField(WebPage*, WebCore::HTMLInputElement&, InputFieldAction, WebFrame*) override;
    void willSubmitForm(WebPage*, WebCore::HTMLFormElement*, WebFrame*, WebFrame* sourceFrame, const Vector<std::pair<String, String>>&, RefPtr<API::Object>& userData) override;
    void willSendSubmitEvent(WebPage*, WebCore::HTMLFormElement*, WebFrame*, WebFrame* sourceFrame, const Vector<std::pair<String, String>>&) override;
    void didAssociateFormControls(WebPage*, const Vector<RefPtr<WebCore::Element>>&, WebFrame*) override;
    bool shouldNotifyOnFormChanges(WebPage*) override;
};

} // namespace WebKit

#endif // InjectedBundlePageFormClient_h
