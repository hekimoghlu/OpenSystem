/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 8, 2022.
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
#import "DOMAbstractViewInternal.h"
#import "DOMCSSRuleInternal.h"
#import "DOMCSSRuleListInternal.h"
#import "DOMCSSStyleDeclarationInternal.h"
#import "DOMCSSValueInternal.h"
#import "DOMCounterInternal.h"
#import "DOMEventInternal.h"
#import "DOMHTMLCollectionInternal.h"
#import "DOMImplementationInternal.h"
#import "DOMInternal.h"
#import "DOMMediaListInternal.h"
#import "DOMNamedNodeMapInternal.h"
#import "DOMNodeInternal.h"
#import "DOMNodeIteratorInternal.h"
#import "DOMNodeListInternal.h"
#import "DOMRGBColorInternal.h"
#import "DOMRangeInternal.h"
#import "DOMRectInternal.h"
#import "DOMStyleSheetInternal.h"
#import "DOMStyleSheetListInternal.h"
#import "DOMTreeWalkerInternal.h"
#import "DOMXPathExpressionInternal.h"
#import "DOMXPathResultInternal.h"
#import <WebCore/JSCSSRule.h>
#import <WebCore/JSCSSRuleList.h>
#import <WebCore/JSCSSStyleDeclaration.h>
#import <WebCore/JSDOMImplementation.h>
#import <WebCore/JSDeprecatedCSSOMCounter.h>
#import <WebCore/JSDeprecatedCSSOMRGBColor.h>
#import <WebCore/JSDeprecatedCSSOMRect.h>
#import <WebCore/JSDeprecatedCSSOMValue.h>
#import <WebCore/JSEvent.h>
#import <WebCore/JSHTMLCollection.h>
#import <WebCore/JSHTMLOptionsCollection.h>
#import <WebCore/JSMediaList.h>
#import <WebCore/JSNamedNodeMap.h>
#import <WebCore/JSNode.h>
#import <WebCore/JSNodeIterator.h>
#import <WebCore/JSNodeList.h>
#import <WebCore/JSRange.h>
#import <WebCore/JSStyleSheet.h>
#import <WebCore/JSStyleSheetList.h>
#import <WebCore/JSTreeWalker.h>
#import <WebCore/JSWindowProxy.h>
#import <WebCore/JSXPathExpression.h>
#import <WebCore/JSXPathResult.h>
#import <WebCore/SimpleRange.h>
#import <WebCore/WebScriptObjectPrivate.h>

static WebScriptObject *createDOMWrapper(JSC::JSObject& jsWrapper)
{
    JSC::VM& vm = jsWrapper.vm();
    #define WRAP(className) \
        if (auto* wrapped = WebCore::JS##className::toWrapped(vm, &jsWrapper)) \
            return kit(wrapped);

    WRAP(CSSRule)
    WRAP(CSSRuleList)
    WRAP(CSSStyleDeclaration)
    WRAP(DeprecatedCSSOMValue)
    WRAP(DeprecatedCSSOMCounter)
    WRAP(DOMImplementation)
    WRAP(Event)
    WRAP(HTMLOptionsCollection)
    WRAP(MediaList)
    WRAP(NamedNodeMap)
    WRAP(Node)
    WRAP(NodeIterator)
    WRAP(NodeList)
    WRAP(DeprecatedCSSOMRGBColor)
    WRAP(Range)
    WRAP(DeprecatedCSSOMRect)
    WRAP(StyleSheet)
    WRAP(StyleSheetList)
    WRAP(TreeWalker)
    WRAP(WindowProxy)
    WRAP(XPathExpression)
    WRAP(XPathResult)

    // This must be after HTMLOptionsCollection, because JSHTMLCollection is a base class of
    // JSHTMLOptionsCollection, and HTMLCollection is *not* a base class of HTMLOptionsCollection.
    WRAP(HTMLCollection)

    #undef WRAP

    return nil;
}

static void disconnectWindowWrapper(WebScriptObject *windowWrapper)
{
    ASSERT([windowWrapper isKindOfClass:[DOMAbstractView class]]);
    [(DOMAbstractView *)windowWrapper _disconnectFrame];
}

void initializeDOMWrapperHooks()
{
    WebCore::initializeDOMWrapperHooks(createDOMWrapper, disconnectWindowWrapper);
}
