/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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

#include "ExceptionOr.h"
#include "JSDOMGuardedObject.h"
#include <JavaScriptCore/JSObject.h>

namespace WebCore {
class ReadableStreamSink;

class InternalReadableStream final : public DOMGuarded<JSC::JSObject> {
public:
    static ExceptionOr<Ref<InternalReadableStream>> createFromUnderlyingSource(JSDOMGlobalObject&, JSC::JSValue underlyingSink, JSC::JSValue strategy);
    static Ref<InternalReadableStream> fromObject(JSDOMGlobalObject&, JSC::JSObject&);

    operator JSC::JSValue() const { return guarded(); }

    void lock();
    bool isLocked() const;
    WEBCORE_EXPORT bool isDisturbed() const;
    void cancel(Exception&&);
    void pipeTo(ReadableStreamSink&);
    ExceptionOr<std::pair<Ref<InternalReadableStream>, Ref<InternalReadableStream>>> tee(bool shouldClone);

    JSC::JSValue cancelForBindings(JSC::JSGlobalObject& globalObject, JSC::JSValue value) { return cancel(globalObject, value, Use::Bindings); }
    JSC::JSValue getReader(JSC::JSGlobalObject&, JSC::JSValue);
    JSC::JSValue pipeTo(JSC::JSGlobalObject&, JSC::JSValue, JSC::JSValue);
    JSC::JSValue pipeThrough(JSC::JSGlobalObject&, JSC::JSValue, JSC::JSValue);

private:
    InternalReadableStream(JSDOMGlobalObject& globalObject, JSC::JSObject& jsObject)
        : DOMGuarded<JSC::JSObject>(globalObject, jsObject)
    {
    }

    enum class Use { Bindings, Private };
    JSC::JSValue cancel(JSC::JSGlobalObject&, JSC::JSValue, Use);
    JSC::JSValue tee(JSC::JSGlobalObject&, bool shouldClone);
};

}
