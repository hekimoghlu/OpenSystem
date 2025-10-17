/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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
#import "DOMDocumentFragmentInternal.h"
#import "DOMExtensions.h"
#import "DOMHTMLCollectionInternal.h"
#import "DOMHTMLDocumentInternal.h"
#import "DOMHTMLInputElementInternal.h"
#import "DOMHTMLSelectElementInternal.h"
#import "DOMHTMLTextAreaElementInternal.h"
#import "DOMNodeInternal.h"
#import "DOMPrivate.h"
#import <WebCore/DocumentFragment.h>
#import <WebCore/HTMLCollectionInlines.h>
#import <WebCore/HTMLDocument.h>
#import <WebCore/HTMLInputElement.h>
#import <WebCore/HTMLSelectElement.h>
#import <WebCore/HTMLTextAreaElement.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/Range.h>
#import <WebCore/RenderTextControl.h>
#import <WebCore/Settings.h>
#import <WebCore/SimpleRange.h>
#import <WebCore/markup.h>

#if PLATFORM(IOS_FAMILY)
#import "DOMHTMLElementInternal.h"
#import <WebCore/Autocapitalize.h>
#import <WebCore/HTMLTextFormControlElement.h>
#import <WebCore/JSExecState.h>
#import <WebCore/RenderLayer.h>
#import <WebCore/RenderLayerScrollableArea.h>
#import <WebCore/WAKWindow.h>
#import <WebCore/WebCoreThreadMessage.h>
#endif

// FIXME: We should move all these into the various specific element source files.
// These were originally here because they were hand written and the rest generated,
// but that is no longer true.

#if PLATFORM(IOS_FAMILY)

@implementation DOMHTMLElement (DOMHTMLElementExtensions)

- (int)scrollXOffset
{
    auto* renderer = core(self)->renderer();
    if (!renderer)
        return 0;

    if (!is<WebCore::RenderBlockFlow>(*renderer))
        renderer = renderer->containingBlock();

    if (!is<WebCore::RenderBox>(*renderer) || !renderer->hasNonVisibleOverflow())
        return 0;

    auto* layer = downcast<WebCore::RenderBox>(*renderer).layer();
    if (!layer)
        return 0;
    auto* scrollableArea = layer->scrollableArea();
    if (!scrollableArea)
        return 0;

    return scrollableArea->scrollOffset().x();
}

- (int)scrollYOffset
{
    auto* renderer = core(self)->renderer();
    if (!renderer)
        return 0;

    if (!is<WebCore::RenderBlockFlow>(*renderer))
        renderer = renderer->containingBlock();
    if (!is<WebCore::RenderBox>(*renderer) || !renderer->hasNonVisibleOverflow())
        return 0;

    auto* layer = downcast<WebCore::RenderBox>(*renderer).layer();
    if (!layer)
        return 0;
    auto* scrollableArea = layer->scrollableArea();
    if (!scrollableArea)
        return 0;

    return scrollableArea->scrollOffset().y();
}

- (void)setScrollXOffset:(int)x scrollYOffset:(int)y
{
    [self setScrollXOffset:x scrollYOffset:y adjustForIOSCaret:NO];
}

- (void)setScrollXOffset:(int)x scrollYOffset:(int)y adjustForIOSCaret:(BOOL)adjustForIOSCaret
{
    auto* renderer = core(self)->renderer();
    if (!renderer)
        return;

    if (!is<WebCore::RenderBlockFlow>(*renderer))
        renderer = renderer->containingBlock();
    if (!renderer->hasNonVisibleOverflow() || !is<WebCore::RenderBox>(*renderer))
        return;

    auto* layer = downcast<WebCore::RenderBox>(*renderer).layer();
    if (!layer)
        return;
    auto* scrollableArea = layer->ensureLayerScrollableArea();

    auto scrollPositionChangeOptions = WebCore::ScrollPositionChangeOptions::createProgrammatic();
    scrollPositionChangeOptions.clamping = WebCore::ScrollClamping::Unclamped;
    scrollableArea->scrollToOffset(WebCore::ScrollOffset(x, y), scrollPositionChangeOptions);
}

- (void)absolutePosition:(int *)x :(int *)y :(int *)w :(int *)h
{
    auto* renderer = core(self)->renderBox();
    if (renderer) {
        if (w)
            *w = renderer->width();
        if (h)
            *h = renderer->width();
        if (x && y) {
            WebCore::FloatPoint floatPoint(*x, *y);
            renderer->localToAbsolute(floatPoint);
            WebCore::IntPoint point = roundedIntPoint(floatPoint);
            *x = point.x();
            *y = point.y();
        }
    }
}

@end

#endif // PLATFORM(IOS_FAMILY)

//------------------------------------------------------------------------------------------
// DOMHTMLDocument

@implementation DOMHTMLDocument (DOMHTMLDocumentExtensions)

- (DOMDocumentFragment *)createDocumentFragmentWithMarkupString:(NSString *)markupString baseURL:(NSURL *)baseURL
{
    return kit(createFragmentFromMarkup(*core(self), markupString, [baseURL absoluteString]).ptr());
}

- (DOMDocumentFragment *)createDocumentFragmentWithText:(NSString *)text
{
    // FIXME: Since this is not a contextual fragment, it won't handle whitespace properly.
    return kit(createFragmentFromText(makeRangeSelectingNodeContents(*core(self)), text).ptr());
}

@end

@implementation DOMHTMLDocument (WebPrivate)

- (DOMDocumentFragment *)_createDocumentFragmentWithMarkupString:(NSString *)markupString baseURLString:(NSString *)baseURLString
{
    NSURL *baseURL = core(self)->completeURL(baseURLString);
    return [self createDocumentFragmentWithMarkupString:markupString baseURL:baseURL];
}

- (DOMDocumentFragment *)_createDocumentFragmentWithText:(NSString *)text
{
    return [self createDocumentFragmentWithText:text];
}

@end

@implementation DOMHTMLInputElement (FormAutoFillTransition)

- (BOOL)_isTextField
{
    return core(self)->isTextField();
}

@end

@implementation DOMHTMLSelectElement (FormAutoFillTransition)

- (void)_activateItemAtIndex:(int)index
{
    // Use the setSelectedIndexByUser function so a change event will be fired. <rdar://problem/6760590>
    if (WebCore::HTMLSelectElement* select = core(self))
        select->optionSelectedByUser(index, true);
}

- (void)_activateItemAtIndex:(int)index allowMultipleSelection:(BOOL)allowMultipleSelection
{
    // Use the setSelectedIndexByUser function so a change event will be fired. <rdar://problem/6760590>
    // If this is a <select multiple> the allowMultipleSelection flag will allow setting multiple
    // selections without clearing the other selections.
    if (WebCore::HTMLSelectElement* select = core(self))
        select->optionSelectedByUser(index, true, allowMultipleSelection);
}

@end

#if PLATFORM(IOS_FAMILY)

@implementation DOMHTMLInputElement (FormPromptAdditions)

- (BOOL)_isEdited
{
    return core(self)->lastChangeWasUserEdit();
}

@end

@implementation DOMHTMLTextAreaElement (FormPromptAdditions)

- (BOOL)_isEdited
{
    return core(self)->lastChangeWasUserEdit();
}

@end

static WebAutocapitalizeType webAutocapitalizeType(WebCore::AutocapitalizeType type)
{
    switch (type) {
    case WebCore::AutocapitalizeType::Default:
        return WebAutocapitalizeTypeDefault;
    case WebCore::AutocapitalizeType::None:
        return WebAutocapitalizeTypeNone;
    case WebCore::AutocapitalizeType::Words:
        return WebAutocapitalizeTypeWords;
    case WebCore::AutocapitalizeType::Sentences:
        return WebAutocapitalizeTypeSentences;
    case WebCore::AutocapitalizeType::AllCharacters:
        return WebAutocapitalizeTypeAllCharacters;
    }
}

@implementation DOMHTMLInputElement (AutocapitalizeAdditions)

- (WebAutocapitalizeType)_autocapitalizeType
{
    WebCore::HTMLInputElement* inputElement = core(self);
    return webAutocapitalizeType(inputElement->autocapitalizeType());
}

@end

@implementation DOMHTMLTextAreaElement (AutocapitalizeAdditions)

- (WebAutocapitalizeType)_autocapitalizeType
{
    WebCore::HTMLTextAreaElement* textareaElement = core(self);
    return webAutocapitalizeType(textareaElement->autocapitalizeType());
}

@end

@implementation DOMHTMLInputElement (WebInputChangeEventAdditions)

- (void)setValueWithChangeEvent:(NSString *)newValue
{
    WebCore::JSMainThreadNullState state;
    core(self)->setValue(newValue, WebCore::DispatchInputAndChangeEvent);
}

- (void)setValueAsNumberWithChangeEvent:(double)newValueAsNumber
{
    WebCore::JSMainThreadNullState state;
    core(self)->setValueAsNumber(newValueAsNumber, WebCore::DispatchInputAndChangeEvent);
}

@end

#endif

Class kitClass(WebCore::HTMLCollection* collection)
{
    if (collection->type() == WebCore::CollectionType::SelectOptions)
        return [DOMHTMLOptionsCollection class];
    return [DOMHTMLCollection class];
}
