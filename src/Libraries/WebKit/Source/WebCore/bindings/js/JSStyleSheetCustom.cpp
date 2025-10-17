/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#include "JSStyleSheet.h"

#include "JSCSSStyleSheet.h"
#include "WebCoreOpaqueRootInlines.h"

namespace WebCore {

template<typename Visitor>
void JSStyleSheet::visitAdditionalChildren(Visitor& visitor)
{
    addWebCoreOpaqueRoot(visitor, wrapped());
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSStyleSheet);

JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<StyleSheet>&& styleSheet)
{
    if (styleSheet->isCSSStyleSheet())
        return createWrapper<CSSStyleSheet>(globalObject, WTFMove(styleSheet));
    return createWrapper<StyleSheet>(globalObject, WTFMove(styleSheet));
}

JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, StyleSheet& stylesheet)
{
    return wrap(lexicalGlobalObject, globalObject, stylesheet);
}

} // namespace WebCore
