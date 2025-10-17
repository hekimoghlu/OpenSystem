/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 8, 2023.
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
#include "JSXMLHttpRequest.h"

#include "JSBlob.h"
#include "JSDOMConvertBufferSource.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertJSON.h"
#include "JSDOMConvertNullable.h"
#include "JSDOMConvertStrings.h"
#include "JSDocument.h"
#include "WebCoreOpaqueRootInlines.h"
#include "XMLHttpRequestUpload.h"

namespace WebCore {
using namespace JSC;

template<typename Visitor>
void JSXMLHttpRequest::visitAdditionalChildren(Visitor& visitor)
{
    if (auto* upload = wrapped().optionalUpload())
        addWebCoreOpaqueRoot(visitor, *upload);

    if (auto* responseDocument = wrapped().optionalResponseXML())
        addWebCoreOpaqueRoot(visitor, *responseDocument);
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSXMLHttpRequest);

JSValue JSXMLHttpRequest::response(JSGlobalObject& lexicalGlobalObject) const
{
    auto cacheResult = [&] (JSValue value) -> JSValue {
        m_response.set(lexicalGlobalObject.vm(), this, value);
        return value;
    };


    if (wrapped().responseCacheIsValid())
        return m_response.get();

    auto type = wrapped().responseType();

    switch (type) {
    case XMLHttpRequest::ResponseType::EmptyString:
    case XMLHttpRequest::ResponseType::Text: {
        auto scope = DECLARE_THROW_SCOPE(lexicalGlobalObject.vm());
        return cacheResult(toJS<IDLNullable<IDLUSVString>>(lexicalGlobalObject, scope, wrapped().responseText()));
    }
    default:
        break;
    }

    if (!wrapped().doneWithoutErrors())
        return cacheResult(jsNull());

    JSValue value;
    switch (type) {
    case XMLHttpRequest::ResponseType::EmptyString:
    case XMLHttpRequest::ResponseType::Text:
        ASSERT_NOT_REACHED();
        return jsUndefined();

    case XMLHttpRequest::ResponseType::Json:
        value = toJS<IDLJSON>(*globalObject(), wrapped().responseTextIgnoringResponseType());
        if (!value)
            value = jsNull();
        break;

    case XMLHttpRequest::ResponseType::Document: {
        auto document = wrapped().responseXML();
        ASSERT(!document.hasException());
        value = toJS<IDLInterface<Document>>(lexicalGlobalObject, *globalObject(), document.releaseReturnValue());
        break;
    }

    case XMLHttpRequest::ResponseType::Blob:
        value = toJSNewlyCreated<IDLInterface<Blob>>(lexicalGlobalObject, *globalObject(), wrapped().createResponseBlob());
        break;

    case XMLHttpRequest::ResponseType::Arraybuffer:
        value = toJS<IDLInterface<ArrayBuffer>>(lexicalGlobalObject, *globalObject(), wrapped().createResponseArrayBuffer());
        break;
    }

    wrapped().didCacheResponse();
    return cacheResult(value);
}

} // namespace WebCore
