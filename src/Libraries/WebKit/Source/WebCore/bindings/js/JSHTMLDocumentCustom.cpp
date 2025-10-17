/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include "JSHTMLDocument.h"


namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, Ref<HTMLDocument>&& passedDocument)
{
    auto& document = passedDocument.get();
    auto* wrapper = createWrapper<HTMLDocument>(globalObject, WTFMove(passedDocument));
    reportMemoryForDocumentIfFrameless(*lexicalGlobalObject, document);
    return wrapper;
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, HTMLDocument& document)
{
    if (auto* wrapper = cachedDocumentWrapper(*lexicalGlobalObject, *globalObject, document))
        return wrapper;
    return toJSNewlyCreated(lexicalGlobalObject, globalObject, Ref<HTMLDocument>(document));
}

} // namespace WebCore
