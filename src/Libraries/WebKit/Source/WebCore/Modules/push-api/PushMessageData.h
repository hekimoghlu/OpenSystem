/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#include <JavaScriptCore/ArrayBuffer.h>
#include <JavaScriptCore/JSCJSValue.h>
#include <JavaScriptCore/Uint8Array.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class Blob;
class JSDOMGlobalObject;
class ScriptExecutionContext;

class PushMessageData final : public RefCounted<PushMessageData> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(PushMessageData);
public:
    static Ref<PushMessageData> create(Vector<uint8_t>&& data) { return adoptRef(*new PushMessageData(WTFMove(data))); }

    ExceptionOr<Ref<JSC::ArrayBuffer>> arrayBuffer();
    Ref<Blob> blob(ScriptExecutionContext&);
    ExceptionOr<Ref<JSC::Uint8Array>> bytes();
    ExceptionOr<JSC::JSValue> json(JSDOMGlobalObject&);
    String text();

private:
    explicit PushMessageData(Vector<uint8_t>&&);

    Vector<uint8_t> m_data;
};

inline PushMessageData::PushMessageData(Vector<uint8_t>&& data)
    : m_data(WTFMove(data))
{
}

} // namespace WebCore
