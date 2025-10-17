/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 10, 2025.
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
class InternalWritableStream final : public DOMGuarded<JSC::JSObject> {
public:
    static ExceptionOr<Ref<InternalWritableStream>> createFromUnderlyingSink(JSDOMGlobalObject&, JSC::JSValue underlyingSink, JSC::JSValue strategy);
    static Ref<InternalWritableStream> fromObject(JSDOMGlobalObject&, JSC::JSObject&);

    operator JSC::JSValue() const { return guarded(); }

    bool locked() const;
    void lock();
    JSC::JSValue abortForBindings(JSC::JSGlobalObject&, JSC::JSValue);
    JSC::JSValue closeForBindings(JSC::JSGlobalObject&);
    ExceptionOr<JSC::JSValue> writeChunkForBingings(JSC::JSGlobalObject&, JSC::JSValue);
    JSC::JSValue getWriter(JSC::JSGlobalObject&);

    void closeIfPossible();

private:
    InternalWritableStream(JSDOMGlobalObject& globalObject, JSC::JSObject& jsObject)
        : DOMGuarded<JSC::JSObject>(globalObject, jsObject)
    {
    }
};

}
