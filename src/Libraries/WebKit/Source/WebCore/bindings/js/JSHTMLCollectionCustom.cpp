/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#include "JSHTMLCollection.h"

#include "HTMLCollectionInlines.h"
#include "JSDOMBinding.h"
#include "JSHTMLAllCollection.h"
#include "JSHTMLFormControlsCollection.h"
#include "JSHTMLOptionsCollection.h"


namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<HTMLCollection>&& collection)
{
    switch (collection->type()) {
    case CollectionType::FormControls:
        return createWrapper<HTMLFormControlsCollection>(globalObject, WTFMove(collection));
    case CollectionType::SelectOptions:
        return createWrapper<HTMLOptionsCollection>(globalObject, WTFMove(collection));
    case CollectionType::DocAll:
        return createWrapper<HTMLAllCollection>(globalObject, WTFMove(collection));
    default:
        break;
    }

    return createWrapper<HTMLCollection>(globalObject, WTFMove(collection));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, HTMLCollection& collection)
{
    return wrap(lexicalGlobalObject, globalObject, collection);
}

} // namespace WebCore
