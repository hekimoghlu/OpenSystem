/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 27, 2023.
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

#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace JSC {
class AbstractSlotVisitor;
class JSObject;
class SlotVisitor;
}

namespace WebCore {

class ScriptExecutionContext;
class Event;
class EventTarget;

class EventListener : public RefCountedAndCanMakeWeakPtr<EventListener> {
public:
    enum Type : uint8_t {
        JSEventListenerType,
        ImageEventListenerType,
        ObjCEventListenerType,
        CPPEventListenerType,
        ConditionEventListenerType,
        GObjectEventListenerType,
        NativeEventListenerType,
        SVGTRefTargetEventListenerType,
        PDFDocumentEventListenerType,
    };

    virtual ~EventListener() = default;
    virtual bool operator==(const EventListener& other) const { return this == &other; }

    virtual void handleEvent(ScriptExecutionContext&, Event&) = 0;

    virtual void visitJSFunction(JSC::AbstractSlotVisitor&) { }
    virtual void visitJSFunction(JSC::SlotVisitor&) { }

    virtual bool isAttribute() const { return false; }
    Type type() const { return m_type; }

#if ASSERT_ENABLED
    virtual void checkValidityForEventTarget(EventTarget&) { }
#endif

    virtual JSC::JSObject* jsFunction() const { return nullptr; }
    virtual JSC::JSObject* wrapper() const { return nullptr; }

protected:
    explicit EventListener(Type type)
        : m_type(type)
    {
    }

private:
    Type m_type;
};

} // namespace WebCore
