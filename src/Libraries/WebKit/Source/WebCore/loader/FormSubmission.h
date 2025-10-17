/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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

#include "FormState.h"
#include "FrameLoaderTypes.h"
#include "ReferrerPolicy.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Event;
class FormData;
class FrameLoadRequest;
class HTMLFormControlElement;

class FormSubmission : public RefCountedAndCanMakeWeakPtr<FormSubmission> {
public:
    enum class Method : uint8_t { Get, Post, Dialog };

    class Attributes {
    public:
        Method method() const { return m_method; }
        static Method parseMethodType(const String&);
        void updateMethodType(const String&);
        static ASCIILiteral methodString(Method);

        const String& action() const { return m_action; }
        void parseAction(const String&);

        const AtomString& target() const { return m_target; }
        void setTarget(const AtomString& target) { m_target = target; }

        const String& encodingType() const { return m_encodingType; }
        static String parseEncodingType(const String&);
        void updateEncodingType(const String&);
        bool isMultiPartForm() const { return m_isMultiPartForm; }

        const String& acceptCharset() const { return m_acceptCharset; }
        void setAcceptCharset(const String& value) { m_acceptCharset = value; }

    private:
        Method m_method { Method::Get };
        bool m_isMultiPartForm { false };
        String m_action;
        AtomString m_target;
        String m_encodingType { "application/x-www-form-urlencoded"_s };
        String m_acceptCharset;
    };

    static Ref<FormSubmission> create(HTMLFormElement&, HTMLFormControlElement* overrideSubmitter, const Attributes&, Event*, LockHistory, FormSubmissionTrigger);

    void populateFrameLoadRequest(FrameLoadRequest&);
    URL requestURL() const;

    Method method() const { return m_method; }
    const URL& action() const { return m_action; }
    const AtomString& target() const { return m_target; }
    const String& contentType() const { return m_contentType; }
    FormState& state() const { return *m_formState; }
    Ref<FormState> takeState() { return m_formState.releaseNonNull(); }
    FormData& data() const { return *m_formData; }
    const String boundary() const { return m_boundary; }
    LockHistory lockHistory() const { return m_lockHistory; }
    Event* event() const { return m_event.get(); }
    RefPtr<Event> protectedEvent() const;
    const String& referrer() const { return m_referrer; }
    const String& origin() const { return m_origin; }

    const String& returnValue() const { return m_returnValue; }

    void clearTarget() { m_target = { }; }
    void setReferrer(const String& referrer) { m_referrer = referrer; }
    void setOrigin(const String& origin) { m_origin = origin; }

    void cancel() { m_wasCancelled = true; }
    bool wasCancelled() const { return m_wasCancelled; }

    NewFrameOpenerPolicy newFrameOpenerPolicy() const { return m_newFrameOpenerPolicy; }
    void setNewFrameOpenerPolicy(NewFrameOpenerPolicy newFrameOpenerPolicy) { m_newFrameOpenerPolicy = newFrameOpenerPolicy; }

    ReferrerPolicy referrerPolicy() const { return m_referrerPolicy; }
    void setReferrerPolicy(ReferrerPolicy referrerPolicy) { m_referrerPolicy = referrerPolicy; }

private:
    // dialog form submissions
    FormSubmission(Method, const String& returnValue, const URL& action, const AtomString& target, const String& contentType, LockHistory, Event*);

    // get/post form submissions
    FormSubmission(Method, const URL& action, const AtomString& target, const String& contentType, Ref<FormState>&&, Ref<FormData>&&, const String& boundary, LockHistory, Event*);

    // FIXME: Hold an instance of Attributes instead of individual members.
    Method m_method;
    bool m_wasCancelled { false };
    URL m_action;
    AtomString m_target;
    String m_contentType;
    RefPtr<FormState> m_formState;
    RefPtr<FormData> m_formData;
    String m_boundary;
    LockHistory m_lockHistory;
    RefPtr<Event> m_event;
    String m_referrer;
    String m_origin;

    String m_returnValue; // for form[method=dialog]

    NewFrameOpenerPolicy m_newFrameOpenerPolicy { NewFrameOpenerPolicy::Allow };
    ReferrerPolicy m_referrerPolicy { ReferrerPolicy::EmptyString };
};

} // namespace WebCore
