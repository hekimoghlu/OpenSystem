/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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

#include "Event.h"
#include "JSValueInWrappedObject.h"
#include "SerializedScriptValue.h"
#include <JavaScriptCore/Strong.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ErrorEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ErrorEvent);
public:
    static Ref<ErrorEvent> create(const String& message, const String& fileName, unsigned lineNumber, unsigned columnNumber, JSC::Strong<JSC::Unknown> error)
    {
        return adoptRef(*new ErrorEvent(message, fileName, lineNumber, columnNumber, error));
    }

    static Ref<ErrorEvent> create(const AtomString& type, const String& message, const String& fileName, unsigned lineNumber, unsigned columnNumber, JSC::Strong<JSC::Unknown> error)
    {
        return adoptRef(*new ErrorEvent(type, message, fileName, lineNumber, columnNumber, error));
    }

    struct Init : EventInit {
        String message;
        String filename;
        unsigned lineno { 0 };
        unsigned colno { 0 };
        JSC::JSValue error { JSC::jsUndefined() };
    };

    static Ref<ErrorEvent> create(const AtomString& type, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new ErrorEvent(type, initializer, isTrusted));
    }

    virtual ~ErrorEvent();

    const String& message() const { return m_message; }
    const String& filename() const { return m_fileName; }
    unsigned lineno() const { return m_lineNumber; }
    unsigned colno() const { return m_columnNumber; }
    JSC::JSValue error(JSC::JSGlobalObject&);

    const JSValueInWrappedObject& originalError() const { return m_error; }
    SerializedScriptValue* serializedError() const { return m_serializedError.get(); }

    RefPtr<SerializedScriptValue> trySerializeError(JSC::JSGlobalObject&);

private:
    ErrorEvent(const AtomString& type, const String& message, const String& fileName, unsigned lineNumber, unsigned columnNumber, JSC::Strong<JSC::Unknown> error);
    ErrorEvent(const String& message, const String& fileName, unsigned lineNumber, unsigned columnNumber, JSC::Strong<JSC::Unknown> error);
    ErrorEvent(const AtomString&, const Init&, IsTrusted);

    bool isErrorEvent() const override;

    String m_message;
    String m_fileName;
    unsigned m_lineNumber;
    unsigned m_columnNumber;
    JSValueInWrappedObject m_error;
    RefPtr<SerializedScriptValue> m_serializedError;
    bool m_triedToSerialize { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(ErrorEvent)
