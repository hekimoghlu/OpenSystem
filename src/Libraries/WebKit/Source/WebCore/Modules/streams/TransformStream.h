/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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
#include "JSDOMGlobalObject.h"
#include "JSValueInWrappedObject.h"
#include <JavaScriptCore/Strong.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class InternalReadableStream;
class InternalTransformStream;
class InternalWritableStream;
class JSDOMGlobalObject;
class ReadableStream;
class WritableStream;

class TransformStream : public RefCounted<TransformStream> {
public:
    static ExceptionOr<Ref<TransformStream>> create(JSC::JSGlobalObject&, std::optional<JSC::Strong<JSC::JSObject>>&&, std::optional<JSC::Strong<JSC::JSObject>>&&, std::optional<JSC::Strong<JSC::JSObject>>&&);

    ~TransformStream();

    ReadableStream& readable() { return m_readable.get(); }
    WritableStream& writable() { return m_writable.get(); }

    JSValueInWrappedObject& internalTransformStream() { return m_internalTransformStream; }

private:
    TransformStream(JSC::JSValue, Ref<ReadableStream>&&, Ref<WritableStream>&&);

    JSValueInWrappedObject m_internalTransformStream;
    Ref<ReadableStream> m_readable;
    Ref<WritableStream> m_writable;
};

} // namespace WebCore
