/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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

#include "JSDOMConvertBufferSource.h"
#include "JSReadableStreamDefaultController.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/TypedArrays.h>

namespace WebCore {

class ReadableStreamSource;

class ReadableStreamDefaultController {
public:
    explicit ReadableStreamDefaultController(JSReadableStreamDefaultController* controller) : m_jsController(controller) { }

    WEBCORE_EXPORT bool enqueue(RefPtr<JSC::ArrayBuffer>&&);
    bool enqueue(JSC::JSValue);
    WEBCORE_EXPORT void error(const Exception&);
    WEBCORE_EXPORT void close();

private:
    JSReadableStreamDefaultController& jsController() const;

    JSDOMGlobalObject& globalObject() const;

    // The owner of ReadableStreamDefaultController is responsible to keep uncollected the JSReadableStreamDefaultController.
    JSReadableStreamDefaultController* m_jsController { nullptr };
};

inline JSReadableStreamDefaultController& ReadableStreamDefaultController::jsController() const
{
    ASSERT(m_jsController);
    return *m_jsController;
}

inline JSDOMGlobalObject& ReadableStreamDefaultController::globalObject() const
{
    ASSERT(m_jsController);
    ASSERT(m_jsController->globalObject());
    return *static_cast<JSDOMGlobalObject*>(m_jsController->globalObject());
}

} // namespace WebCore
