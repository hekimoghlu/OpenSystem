/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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
#include "JSHTMLAllCollection.h"

#include "CachedHTMLCollectionInlines.h"
#include "Element.h"
#include "HTMLCollection.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertNullable.h"
#include "JSDOMConvertUnion.h"
#include "JSElement.h"
#include "JSHTMLCollection.h"

namespace WebCore {
using namespace JSC;

static JSC_DECLARE_HOST_FUNCTION(callJSHTMLAllCollection);

// https://html.spec.whatwg.org/multipage/common-dom-interfaces.html#HTMLAllCollection-call
JSC_DEFINE_HOST_FUNCTION(callJSHTMLAllCollection, (JSGlobalObject* lexicalGlobalObject, CallFrame* callFrame))
{
    VM& vm = lexicalGlobalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* castedThis = jsCast<JSHTMLAllCollection*>(callFrame->jsCallee());
    ASSERT(castedThis);
    auto& impl = castedThis->wrapped();
    if (callFrame->argument(0).isUndefined())
        return JSValue::encode(jsNull());

    AtomString nameOrIndex = callFrame->uncheckedArgument(0).toString(lexicalGlobalObject)->toAtomString(lexicalGlobalObject);
    RETURN_IF_EXCEPTION(scope, { });
    RELEASE_AND_RETURN(scope, JSValue::encode(toJS<IDLNullable<IDLUnion<IDLInterface<HTMLCollection>, IDLInterface<Element>>>>(*lexicalGlobalObject, *castedThis->globalObject(), impl.namedOrIndexedItemOrItems(WTFMove(nameOrIndex)))));
}

CallData JSHTMLAllCollection::getCallData(JSCell*)
{
    CallData callData;
    callData.type = CallData::Type::Native;
    callData.native.function = callJSHTMLAllCollection;
    callData.native.isBoundFunction = false;
    callData.native.isWasm = false;
    return callData;
}

} // namespace WebCore
