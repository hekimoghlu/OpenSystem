/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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
#include "JSCSSStyleDeclaration.h"

#include "DOMWrapperWorld.h"
#include "JSCSSRuleCustom.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertStrings.h"
#include "JSDeprecatedCSSOMValue.h"
#include "JSNodeCustom.h"
#include "JSStyleSheetCustom.h"
#include "StyledElement.h"
#include "WebCoreOpaqueRootInlines.h"


namespace WebCore {
using namespace JSC;

WebCoreOpaqueRoot root(CSSStyleDeclaration* style)
{
    ASSERT(style);
    if (auto* parentRule = style->parentRule())
        return root(parentRule);
    if (auto* styleSheet = style->parentStyleSheet())
        return root(styleSheet);
    if (auto* parentElement = style->parentElement())
        return root(parentElement);
    return WebCoreOpaqueRoot { style };
}

template<typename Visitor>
void JSCSSStyleDeclaration::visitAdditionalChildren(Visitor& visitor)
{
    addWebCoreOpaqueRoot(visitor, wrapped());
}

DEFINE_VISIT_ADDITIONAL_CHILDREN(JSCSSStyleDeclaration);

} // namespace WebCore
