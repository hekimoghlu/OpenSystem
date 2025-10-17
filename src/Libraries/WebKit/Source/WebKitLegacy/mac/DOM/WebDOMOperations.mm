/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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
#import "WebDOMOperationsPrivate.h"

#import "DOMDocumentInternal.h"
#import "DOMElementInternal.h"
#import "DOMNodeInternal.h"
#import "DOMRangeInternal.h"
#import "DOMWheelEventInternal.h"
#import "WebArchiveInternal.h"
#import "WebDataSourcePrivate.h"
#import "WebFrameInternal.h"
#import "WebFrameLoaderClient.h"
#import "WebFramePrivate.h"
#import "WebKitNSStringExtras.h"
#import <JavaScriptCore/APICast.h>
#import <JavaScriptCore/JSCJSValue.h>
#import <JavaScriptCore/JSGlobalObjectInlines.h>
#import <JavaScriptCore/JSLock.h>
#import <WebCore/Document.h>
#import <WebCore/FrameLoader.h>
#import <WebCore/HTMLInputElement.h>
#import <WebCore/HTMLTextFormControlElement.h>
#import <WebCore/JSElement.h>
#import <WebCore/LegacyWebArchive.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/PlatformWheelEvent.h>
#import <WebCore/Range.h>
#import <WebCore/RenderElement.h>
#import <WebCore/RenderStyleInlines.h>
#import <WebCore/RenderTreeAsText.h>
#import <WebCore/ShadowRoot.h>
#import <WebCore/SimpleRange.h>
#import <WebCore/WheelEvent.h>
#import <WebCore/markup.h>
#import <WebKitLegacy/DOMExtensions.h>
#import <WebKitLegacy/DOMHTML.h>
#import <wtf/Assertions.h>
#import <wtf/text/MakeString.h>

using namespace WebCore;
using namespace JSC;

@implementation DOMElement (WebDOMElementOperationsPrivate)

+ (DOMElement *)_DOMElementFromJSContext:(JSContextRef)context value:(JSValueRef)value
{
    if (!context)
        return 0;

    if (!value)
        return 0;

    JSGlobalObject* lexicalGlobalObject = toJS(context);
    JSLockHolder lock(lexicalGlobalObject);
    return kit(JSElement::toWrapped(lexicalGlobalObject->vm(), toJS(lexicalGlobalObject, value)));
}

@end

@implementation DOMNode (WebDOMNodeOperations)

- (WebArchive *)webArchive
{
    return adoptNS([[WebArchive alloc] _initWithCoreLegacyWebArchive:LegacyWebArchive::create(*core(self))]).autorelease();
}

- (WebArchive *)webArchiveByFilteringSubframes:(WebArchiveSubframeFilter)webArchiveSubframeFilter
{
    auto webArchive = adoptNS([[WebArchive alloc] _initWithCoreLegacyWebArchive:LegacyWebArchive::create(*core(self), [webArchiveSubframeFilter](LocalFrame& subframe) -> bool {
        return webArchiveSubframeFilter(kit(&subframe));
    })]);

    return webArchive.autorelease();
}

#if PLATFORM(IOS_FAMILY)

- (BOOL)isHorizontalWritingMode
{
    Node* node = core(self);
    if (!node)
        return YES;
    
    RenderObject* renderer = node->renderer();
    if (!renderer)
        return YES;
    
    return renderer->writingMode().isHorizontal();
}

- (void)hidePlaceholder
{
    if (auto node = core(self); is<HTMLTextFormControlElement>(node))
        downcast<HTMLTextFormControlElement>(*node).setCanShowPlaceholder(false);
}

- (void)showPlaceholderIfNecessary
{
    if (auto node = core(self); is<HTMLTextFormControlElement>(node))
        downcast<HTMLTextFormControlElement>(*node).setCanShowPlaceholder(true);
}

#endif

@end

@implementation DOMNode (WebDOMNodeOperationsPendingPublic)

- (NSString *)markupString
{
    auto& node = *core(self);

    String markupString = serializeFragment(node, SerializedNodes::SubtreeIncludingNode);
    Node::NodeType nodeType = node.nodeType();
    if (nodeType != Node::DOCUMENT_NODE && nodeType != Node::DOCUMENT_TYPE_NODE)
        markupString = makeString(documentTypeString(node.document()), markupString);

    return markupString;
}

- (NSRect)_renderRect:(bool *)isReplaced
{
    return NSRect(core(self)->pixelSnappedAbsoluteBoundingRect(isReplaced));
}

@end

@implementation DOMDocument (WebDOMDocumentOperations)

- (WebFrame *)webFrame
{
    auto* frame = core(self)->frame();
    if (!frame)
        return nil;
    return kit(frame);
}

- (NSURL *)URLWithAttributeString:(NSString *)string
{
    return core(self)->completeURL(string);
}

@end

@implementation DOMDocument (WebDOMDocumentOperationsInternal)

- (DOMRange *)_documentRange
{
    DOMRange *range = [self createRange];

    if (DOMNode* documentElement = [self documentElement])
        [range selectNode:documentElement];

    return range;
}

@end

@implementation DOMRange (WebDOMRangeOperations)

- (WebArchive *)webArchive
{
    return adoptNS([[WebArchive alloc] _initWithCoreLegacyWebArchive:LegacyWebArchive::create(makeSimpleRange(*core(self)))]).autorelease();
}

- (NSString *)markupString
{
    auto range = makeSimpleRange(*core(self));
    return makeString(documentTypeString(range.start.document()), serializePreservingVisualAppearance(range, nullptr, AnnotateForInterchange::Yes));
}

@end

@implementation DOMHTMLFrameElement (WebDOMHTMLFrameElementOperations)

- (WebFrame *)contentFrame
{
    return [[self contentDocument] webFrame];
}

@end

@implementation DOMHTMLIFrameElement (WebDOMHTMLIFrameElementOperations)

- (WebFrame *)contentFrame
{
    return [[self contentDocument] webFrame];
}

@end

@implementation DOMHTMLInputElement (WebDOMHTMLInputElementOperationsPrivate)

- (BOOL)_isAutofilled
{
    return downcast<HTMLInputElement>(core((DOMElement *)self))->autofilled();
}

- (BOOL)_isAutoFilledAndViewable
{
    return downcast<HTMLInputElement>(core((DOMElement *)self))->autofilledAndViewable();
}

- (void)_setAutofilled:(BOOL)autofilled
{
    downcast<HTMLInputElement>(core((DOMElement *)self))->setAutofilled(autofilled);
}

- (void)_setAutoFilledAndViewable:(BOOL)autoFilledAndViewable
{
    downcast<HTMLInputElement>(core((DOMElement *)self))->setAutofilledAndViewable(autoFilledAndViewable);
}

@end

@implementation DOMHTMLObjectElement (WebDOMHTMLObjectElementOperations)

- (WebFrame *)contentFrame
{
    return [[self contentDocument] webFrame];
}

@end

#if !PLATFORM(IOS_FAMILY)
static NSEventPhase toNSEventPhase(PlatformWheelEventPhase platformPhase)
{
    switch (platformPhase) {
    case PlatformWheelEventPhase::None:
        return NSEventPhaseNone;
    case PlatformWheelEventPhase::Began:
        return NSEventPhaseBegan;
    case PlatformWheelEventPhase::Stationary:
        return NSEventPhaseStationary;
    case PlatformWheelEventPhase::Changed:
        return NSEventPhaseChanged;
    case PlatformWheelEventPhase::Ended:
        return NSEventPhaseEnded;
    case PlatformWheelEventPhase::Cancelled:
        return NSEventPhaseCancelled;
    case PlatformWheelEventPhase::MayBegin:
        return NSEventPhaseMayBegin;
    }

    return NSEventPhaseNone;
}

@implementation DOMWheelEvent (WebDOMWheelEventOperationsPrivate)

- (NSEventPhase)_phase
{
    return toNSEventPhase(core(self)->phase());
}

- (NSEventPhase)_momentumPhase
{
    return toNSEventPhase(core(self)->momentumPhase());
}

@end
#endif
