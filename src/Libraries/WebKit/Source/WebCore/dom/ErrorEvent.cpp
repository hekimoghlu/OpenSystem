/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#include "ErrorEvent.h"

#include "DOMWrapperWorld.h"
#include "EventNames.h"
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/StrongInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
using namespace JSC;

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ErrorEvent);

ErrorEvent::ErrorEvent(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::ErrorEvent, type, initializer, isTrusted)
    , m_message(initializer.message)
    , m_fileName(initializer.filename)
    , m_lineNumber(initializer.lineno)
    , m_columnNumber(initializer.colno)
    , m_error(initializer.error)
{
}

ErrorEvent::ErrorEvent(const AtomString& type, const String& message, const String& fileName, unsigned lineNumber, unsigned columnNumber, JSC::Strong<JSC::Unknown> error)
    : Event(EventInterfaceType::ErrorEvent, type, CanBubble::No, IsCancelable::Yes)
    , m_message(message)
    , m_fileName(fileName)
    , m_lineNumber(lineNumber)
    , m_columnNumber(columnNumber)
    , m_error(error.get())
{
}

ErrorEvent::ErrorEvent(const String& message, const String& fileName, unsigned lineNumber, unsigned columnNumber, JSC::Strong<JSC::Unknown> error)
    : ErrorEvent(eventNames().errorEvent, message, fileName, lineNumber, columnNumber, error)
{
}

ErrorEvent::~ErrorEvent() = default;

JSValue ErrorEvent::error(JSGlobalObject& globalObject)
{    
    if (!m_error)
        return jsNull();

    JSValue error = m_error.getValue();
    if (!isWorldCompatible(globalObject, error)) {
        // We need to make sure ErrorEvents do not leak their error property across isolated DOM worlds.
        // Ideally, we would check that the worlds have different privileges but that's not possible yet.
        auto serializedError = trySerializeError(globalObject);
        if (!serializedError)
            return jsNull();
        return serializedError->deserialize(globalObject, &globalObject);
    }

    return error;
}

RefPtr<SerializedScriptValue> ErrorEvent::trySerializeError(JSGlobalObject& exec)
{
    if (!m_serializedError && !m_triedToSerialize) {
        m_serializedError = SerializedScriptValue::create(exec, m_error.getValue(), SerializationForStorage::No, SerializationErrorMode::NonThrowing);
        m_triedToSerialize = true;
    }
    return m_serializedError;
}

bool ErrorEvent::isErrorEvent() const
{
    return true;
}

} // namespace WebCore
