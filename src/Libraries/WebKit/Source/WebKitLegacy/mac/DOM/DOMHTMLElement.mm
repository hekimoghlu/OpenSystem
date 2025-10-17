/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
#import "DOMHTMLElementInternal.h"

#import "DOMElementInternal.h"
#import "DOMHTMLCollectionInternal.h"
#import "DOMNodeInternal.h"
#import <WebCore/Element.h>
#import "ExceptionHandlers.h"
#import <WebCore/ElementInlines.h>
#import <WebCore/HTMLCollection.h>
#import <WebCore/HTMLElement.h>
#import <WebCore/HTMLNames.h>
#import <WebCore/HitTestResult.h>
#import <WebCore/JSExecState.h>
#import <WebCore/ThreadCheck.h>
#import <WebCore/WebScriptObjectPrivate.h>
#import <wtf/GetPtr.h>
#import <wtf/URL.h>

#define IMPL static_cast<WebCore::HTMLElement*>(reinterpret_cast<WebCore::Node*>(_internal))

@implementation DOMHTMLElement

- (NSString *)title
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::titleAttr);
}

- (void)setTitle:(NSString *)newTitle
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::titleAttr, newTitle);
}

- (NSString *)lang
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::langAttr);
}

- (void)setLang:(NSString *)newLang
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::langAttr, newLang);
}

- (BOOL)translate
{
    WebCore::JSMainThreadNullState state;
    return IMPL->translate();
}

- (void)setTranslate:(BOOL)newTranslate
{
    WebCore::JSMainThreadNullState state;
    IMPL->setTranslate(newTranslate);
}

- (NSString *)dir
{
    WebCore::JSMainThreadNullState state;
    return IMPL->dir();
}

- (void)setDir:(NSString *)newDir
{
    WebCore::JSMainThreadNullState state;
    IMPL->setDir(newDir);
}

- (int)tabIndex
{
    WebCore::JSMainThreadNullState state;
    return IMPL->tabIndexForBindings();
}

- (void)setTabIndex:(int)newTabIndex
{
    WebCore::JSMainThreadNullState state;
    IMPL->setTabIndexForBindings(newTabIndex);
}

- (BOOL)draggable
{
    WebCore::JSMainThreadNullState state;
    return IMPL->draggable();
}

- (void)setDraggable:(BOOL)newDraggable
{
    WebCore::JSMainThreadNullState state;
    IMPL->setDraggable(newDraggable);
}

- (NSString *)webkitdropzone
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::webkitdropzoneAttr);
}

- (void)setWebkitdropzone:(NSString *)newWebkitdropzone
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::webkitdropzoneAttr, newWebkitdropzone);
}

- (BOOL)hidden
{
    WebCore::JSMainThreadNullState state;
    return IMPL->hasAttributeWithoutSynchronization(WebCore::HTMLNames::hiddenAttr);
}

- (void)setHidden:(BOOL)newHidden
{
    WebCore::JSMainThreadNullState state;
    IMPL->setBooleanAttribute(WebCore::HTMLNames::hiddenAttr, newHidden);
}

- (NSString *)accessKey
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getAttribute(WebCore::HTMLNames::accesskeyAttr);
}

- (void)setAccessKey:(NSString *)newAccessKey
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::accesskeyAttr, newAccessKey);
}

- (NSString *)innerText
{
    WebCore::JSMainThreadNullState state;
    return IMPL->innerText();
}

- (void)setInnerText:(NSString *)newInnerText
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setInnerText(newInnerText));
}

- (NSString *)outerText
{
    WebCore::JSMainThreadNullState state;
    return IMPL->outerText();
}

- (void)setOuterText:(NSString *)newOuterText
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setOuterText(newOuterText));
}

- (NSString *)contentEditable
{
    WebCore::JSMainThreadNullState state;
    return IMPL->contentEditable();
}

- (void)setContentEditable:(NSString *)newContentEditable
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->setContentEditable(newContentEditable));
}

- (BOOL)isContentEditable
{
    WebCore::JSMainThreadNullState state;
    return IMPL->isContentEditable();
}

- (BOOL)spellcheck
{
    WebCore::JSMainThreadNullState state;
    return IMPL->spellcheck();
}

- (void)setSpellcheck:(BOOL)newSpellcheck
{
    WebCore::JSMainThreadNullState state;
    IMPL->setSpellcheck(newSpellcheck);
}

- (NSString *)idName
{
    WebCore::JSMainThreadNullState state;
    return IMPL->getIdAttribute();
}

- (void)setIdName:(NSString *)newIdName
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAttributeWithoutSynchronization(WebCore::HTMLNames::idAttr, newIdName);
}

- (DOMHTMLCollection *)children
{
    WebCore::JSMainThreadNullState state;
    return kit(WTF::getPtr(IMPL->children()));
}

- (NSString *)titleDisplayString
{
    WebCore::JSMainThreadNullState state;
    return WebCore::displayString(IMPL->title(), core(self));
}

- (DOMElement *)insertAdjacentElement:(NSString *)where element:(DOMElement *)element
{
    WebCore::JSMainThreadNullState state;
    if (!element)
        raiseTypeErrorException();
    return kit(raiseOnDOMError(IMPL->insertAdjacentElement(where, *core(element))));
}

- (void)insertAdjacentHTML:(NSString *)where html:(NSString *)html
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->insertAdjacentHTML(where, html));
}

- (void)insertAdjacentText:(NSString *)where text:(NSString *)text
{
    WebCore::JSMainThreadNullState state;
    raiseOnDOMError(IMPL->insertAdjacentText(where, text));
}

- (void)click
{
    WebCore::JSMainThreadNullState state;
    IMPL->click();
}

#if ENABLE(AUTOCORRECT)

- (BOOL)autocorrect
{
    WebCore::JSMainThreadNullState state;
    return IMPL->shouldAutocorrect();
}

- (void)setAutocorrect:(BOOL)newAutocorrect
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAutocorrect(newAutocorrect);
}

#endif

#if ENABLE(AUTOCAPITALIZE)

- (NSString *)autocapitalize
{
    WebCore::JSMainThreadNullState state;
    return IMPL->autocapitalize();
}

- (void)setAutocapitalize:(NSString *)newAutocapitalize
{
    WebCore::JSMainThreadNullState state;
    IMPL->setAutocapitalize(newAutocapitalize);
}

#endif

@end

WebCore::HTMLElement* core(DOMHTMLElement *wrapper)
{
    return wrapper ? reinterpret_cast<WebCore::HTMLElement*>(wrapper->_internal) : 0;
}

DOMHTMLElement *kit(WebCore::HTMLElement* value)
{
    WebCoreThreadViolationCheckRoundOne();
    return static_cast<DOMHTMLElement*>(kit(static_cast<WebCore::Node*>(value)));
}

#undef IMPL
