/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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

#include "JSEventListener.h"
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class ContainerNode;
class Document;
class Element;
class LocalDOMWindow;
class QualifiedName;

class JSLazyEventListener final : public JSEventListener {
public:
    static RefPtr<JSLazyEventListener> create(Element&, const QualifiedName& attributeName, const AtomString& attributeValue);
    static RefPtr<JSLazyEventListener> create(Document&, const QualifiedName& attributeName, const AtomString& attributeValue);
    static RefPtr<JSLazyEventListener> create(LocalDOMWindow&, const QualifiedName& attributeName, const AtomString& attributeValue);

    virtual ~JSLazyEventListener();

    URL sourceURL() const final { return m_sourceURL; }
    TextPosition sourcePosition() const final { return m_sourcePosition; }

private:
    struct CreationArguments;
    static RefPtr<JSLazyEventListener> create(CreationArguments&&);
    JSLazyEventListener(CreationArguments&&, const URL& sourceURL, const TextPosition&);
    String code() const final { return m_code; }

#if ASSERT_ENABLED
    void checkValidityForEventTarget(EventTarget&) final;
#endif

    JSC::JSObject* initializeJSFunction(ScriptExecutionContext&) const final;

    String m_functionName;
    const String& m_functionParameters;
    String m_code;
    URL m_sourceURL;
    TextPosition m_sourcePosition;
    WeakPtr<ContainerNode, WeakPtrImplWithEventTargetData> m_originalNode;
    JSC::SourceTaintedOrigin m_sourceTaintedOrigin;
};

} // namespace WebCore
