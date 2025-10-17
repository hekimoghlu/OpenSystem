/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
#ifndef APIInjectedBundleFormClient_h
#define APIInjectedBundleFormClient_h

#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
class Element;
class HTMLFormElement;
class HTMLInputElement;
class HTMLTextAreaElement;
}

namespace WebKit {
class WebFrame;
class WebPage;
}

namespace API {

class Object;

namespace InjectedBundle {

class FormClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(FormClient);
public:
    virtual ~FormClient() { }

    virtual void didFocusTextField(WebKit::WebPage*, WebCore::HTMLInputElement&, WebKit::WebFrame*) { }
    virtual void textFieldDidBeginEditing(WebKit::WebPage*, WebCore::HTMLInputElement&, WebKit::WebFrame*) { }
    virtual void textFieldDidEndEditing(WebKit::WebPage*, WebCore::HTMLInputElement&, WebKit::WebFrame*) { }
    virtual void textDidChangeInTextField(WebKit::WebPage*, WebCore::HTMLInputElement&, WebKit::WebFrame*, bool) { }
    virtual void textDidChangeInTextArea(WebKit::WebPage*, WebCore::HTMLTextAreaElement&, WebKit::WebFrame*) { }

    enum class InputFieldAction {
        MoveUp,
        MoveDown,
        Cancel,
        InsertTab,
        InsertBacktab,
        InsertNewline,
        InsertDelete,
    };

    virtual bool shouldPerformActionInTextField(WebKit::WebPage*, WebCore::HTMLInputElement&, InputFieldAction, WebKit::WebFrame*) { return false; }
    virtual void willSubmitForm(WebKit::WebPage*, WebCore::HTMLFormElement*, WebKit::WebFrame*, WebKit::WebFrame*, const Vector<std::pair<WTF::String, WTF::String>>&, RefPtr<API::Object>& userData) { UNUSED_PARAM(userData); }
    virtual void willSendSubmitEvent(WebKit::WebPage*, WebCore::HTMLFormElement*, WebKit::WebFrame*, WebKit::WebFrame*, const Vector<std::pair<WTF::String, WTF::String>>&) { }
    virtual void didAssociateFormControls(WebKit::WebPage*, const Vector<RefPtr<WebCore::Element>>&, WebKit::WebFrame*) { }
    virtual bool shouldNotifyOnFormChanges(WebKit::WebPage*) { return false; }
    virtual void willBeginInputSession(WebKit::WebPage*, WebCore::Element*, WebKit::WebFrame*, bool userIsInteracting, RefPtr<API::Object>& userData) { UNUSED_PARAM(userData); }
};

} // namespace InjectedBundle

} // namespace API

#endif // APIInjectedBundleFormClient_h
