/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 29, 2021.
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
#include "JSImageData.h"

#include "JSDOMConvert.h"
#include "JSDOMWrapperCache.h"
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/IdentifierInlines.h>
#include <JavaScriptCore/JSObjectInlines.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, Ref<ImageData>&& imageData)
{
    VM& vm = lexicalGlobalObject->vm();
    auto& imageDataArray = imageData->data();
    Ref arrayBufferView = imageDataArray.arrayBufferView();
    auto* wrapper = createWrapper<ImageData>(globalObject, WTFMove(imageData));
    Identifier dataName = Identifier::fromString(vm, "data"_s);
    wrapper->putDirect(vm, dataName, toJS(lexicalGlobalObject, globalObject, arrayBufferView), PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    // FIXME: Adopt reportExtraMemoryVisited, and switch to reportExtraMemoryAllocated.
    // https://bugs.webkit.org/show_bug.cgi?id=142595
    vm.heap.deprecatedReportExtraMemory(imageDataArray.byteLength());
    
    return wrapper;
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, ImageData& imageData)
{
    return wrap(lexicalGlobalObject, globalObject, imageData);
}

}
