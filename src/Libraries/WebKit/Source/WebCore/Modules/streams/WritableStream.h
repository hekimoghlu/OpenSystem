/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#include <JavaScriptCore/Strong.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class InternalWritableStream;
class JSDOMGlobalObject;
class WritableStreamSink;

class WritableStream : public RefCountedAndCanMakeWeakPtr<WritableStream> {
public:
    static ExceptionOr<Ref<WritableStream>> create(JSC::JSGlobalObject&, std::optional<JSC::Strong<JSC::JSObject>>&&, std::optional<JSC::Strong<JSC::JSObject>>&&);
    static ExceptionOr<Ref<WritableStream>> create(JSDOMGlobalObject&, Ref<WritableStreamSink>&&);
    static Ref<WritableStream> create(Ref<InternalWritableStream>&&);

    virtual ~WritableStream();

    void lock();
    bool locked() const;

    void closeIfPossible();

    InternalWritableStream& internalWritableStream();
    enum class Type : bool {
        Default,
        FileSystem
    };
    virtual Type type() const { return Type::Default; }

protected:
    static ExceptionOr<Ref<WritableStream>> create(JSC::JSGlobalObject&, JSC::JSValue, JSC::JSValue);
    static ExceptionOr<Ref<InternalWritableStream>> createInternalWritableStream(JSDOMGlobalObject&, Ref<WritableStreamSink>&&);
    explicit WritableStream(Ref<InternalWritableStream>&&);
private:
    Ref<InternalWritableStream> m_internalWritableStream;
};

} // namespace WebCore
