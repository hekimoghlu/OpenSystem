/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 9, 2024.
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
#include "PushMessageData.h"

#include "Blob.h"
#include "JSDOMGlobalObject.h"
#include "TextResourceDecoder.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <JavaScriptCore/JSLock.h>
#include <JavaScriptCore/JSONObject.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(PushMessageData);

ExceptionOr<Ref<JSC::ArrayBuffer>> PushMessageData::arrayBuffer()
{
    RefPtr buffer = ArrayBuffer::tryCreate(m_data.span());
    if (!buffer)
        return Exception { ExceptionCode::OutOfMemoryError };
    return buffer.releaseNonNull();
}

Ref<Blob> PushMessageData::blob(ScriptExecutionContext& context)
{
    return Blob::create(&context, Vector<uint8_t> { m_data }, { });
}

ExceptionOr<Ref<JSC::Uint8Array>> PushMessageData::bytes()
{
    RefPtr view = Uint8Array::tryCreate(m_data.span());
    if (!view)
        return Exception { ExceptionCode::OutOfMemoryError };
    return view.releaseNonNull();
}

ExceptionOr<JSC::JSValue> PushMessageData::json(JSDOMGlobalObject& globalObject)
{
    JSC::JSLockHolder lock(&globalObject);

    auto value = JSC::JSONParse(&globalObject, text());
    if (!value)
        return Exception { ExceptionCode::SyntaxError, "JSON parsing failed"_s };

    return value;
}

String PushMessageData::text()
{
    return TextResourceDecoder::textFromUTF8(m_data.span());
}

} // namespace WebCore
