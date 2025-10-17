/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
// This header contains WebFrame declarations that can be used anywhere in WebKit, but are neither SPI nor API.

#import "WebFramePrivate.h"
#import "WebPreferencesPrivate.h"
#import <WebCore/CaptionUserPreferences.h>
#import <WebCore/EditAction.h>
#import <WebCore/FrameLoaderTypes.h>
#import <WebCore/FrameSelection.h>
#import <WebCore/Position.h>
#import <WebCore/Settings.h>
#import <wtf/NakedPtr.h>

@class DOMCSSStyleDeclaration;
@class DOMDocumentFragment;
@class DOMElement;
@class DOMNode;
@class DOMRange;
@class WebFrameView;
@class WebHistoryItem;

class WebScriptDebugger;

namespace WebCore {
class CSSStyleDeclaration;
    class Document;
    class DocumentLoader;
    class Element;
    class LocalFrame;
    class HistoryItem;
    class HTMLElement;
    class HTMLFrameOwnerElement;
    class Node;
    class Page;
    class Range;
}

typedef WebCore::HistoryItem WebCoreHistoryItem;

enum class WebRangeIsRelativeTo : uint8_t {
    EditableRoot,
    Paragraph,
};

WebCore::LocalFrame* core(WebFrame *);
WebFrame *kit(WebCore::LocalFrame *);

WebCore::Page* core(WebView *);
WebView *kit(WebCore::Page*);

WebCore::EditableLinkBehavior core(WebKitEditableLinkBehavior);
WebCore::TextDirectionSubmenuInclusionBehavior core(WebTextDirectionSubmenuInclusionBehavior);

#if defined(__cplusplus) && PLATFORM(IOS_FAMILY)
Vector<Vector<String>> vectorForDictationPhrasesArray(NSArray *);
#endif

WebView *getWebView(WebFrame *webFrame);

@interface WebFramePrivate : NSObject {
@public
    NakedPtr<WebCore::LocalFrame> coreFrame;
    RetainPtr<WebFrameView> webFrameView;
    std::unique_ptr<WebScriptDebugger> scriptDebugger;
    std::unique_ptr<WebCore::CaptionUserPreferencesTestingModeToken> captionPreferencesTestingModeToken;
    id internalLoadDelegate;
    BOOL shouldCreateRenderers;
    BOOL includedInWebKitStatistics;
    RetainPtr<NSString> url;
    RetainPtr<NSString> provisionalURL;
#if PLATFORM(IOS_FAMILY)
    BOOL isCommitting;
#endif
}
@end

@protocol WebCoreRenderTreeCopier <NSObject>
- (NSObject *)nodeWithName:(NSString *)name position:(NSPoint)position rect:(NSRect)rect view:(NSView *)view children:(NSArray *)children;
@end

@interface WebFrame (WebInternal)

+ (void)_createMainFrameWithPage:(WebCore::Page*)page frameName:(const WTF::AtomString&)name frameView:(WebFrameView *)frameView;
+ (Ref<WebCore::LocalFrame>)_createSubframeWithOwnerElement:(WebCore::HTMLFrameOwnerElement&)ownerElement page:(WebCore::Page&)page frameName:(const WTF::AtomString&)name frameView:(WebFrameView *)frameView;
- (id)_initWithWebFrameView:(WebFrameView *)webFrameView webView:(WebView *)webView;

- (void)_clearCoreFrame;

- (BOOL)_isIncludedInWebKitStatistics;

- (void)_updateBackgroundAndUpdatesWhileOffscreen;
- (void)_setInternalLoadDelegate:(id)internalLoadDelegate;
- (id)_internalLoadDelegate;
- (void)_unmarkAllBadGrammar;
- (void)_unmarkAllMisspellings;

- (BOOL)_hasSelection;
- (void)_clearSelection;
- (WebFrame *)_findFrameWithSelection;
- (void)_clearSelectionInOtherFrames;

- (void)_attachScriptDebugger;
- (void)_detachScriptDebugger;

// dataSource reports null for the initial empty document's data source; this is needed
// to preserve compatibility with Mail and Safari among others. But internal to WebKit,
// we need to be able to get the initial data source as well, so the _dataSource method
// should be used instead.
- (WebDataSource *)_dataSource;

#if PLATFORM(IOS_FAMILY)
+ (void)_createMainFrameWithSimpleHTMLDocumentWithPage:(WebCore::Page*)page frameView:(WebFrameView *)frameView style:(NSString *)style;

- (BOOL)_isCommitting;
- (void)_setIsCommitting:(BOOL)value;
#endif

- (BOOL)_needsLayout;
- (void)_drawRect:(NSRect)rect contentsOnly:(BOOL)contentsOnly;
- (BOOL)_getVisibleRect:(NSRect*)rect;

- (NSString *)_stringByEvaluatingJavaScriptFromString:(NSString *)string;
- (NSString *)_stringByEvaluatingJavaScriptFromString:(NSString *)string forceUserGesture:(BOOL)forceUserGesture;

- (NSString *)_selectedString;
- (NSString *)_stringForRange:(DOMRange *)range;

- (DOMRange *)_convertNSRangeToDOMRange:(NSRange)range;
- (NSRange)_convertDOMRangeToNSRange:(DOMRange *)range;

- (NSRect)_caretRectAtPosition:(const WebCore::Position&)pos affinity:(NSSelectionAffinity)affinity;
- (NSRect)_firstRectForDOMRange:(DOMRange *)range;
- (void)_scrollDOMRangeToVisible:(DOMRange *)range;
#if PLATFORM(IOS_FAMILY)
- (void)_scrollDOMRangeToVisible:(DOMRange *)range withInset:(CGFloat)inset;
#endif

#if !PLATFORM(IOS_FAMILY)
- (DOMRange *)_rangeByAlteringCurrentSelection:(WebCore::FrameSelection::Alteration)alteration direction:(WebCore::SelectionDirection)direction granularity:(WebCore::TextGranularity)granularity;
#endif

- (NSRange)_convertToNSRange:(const WebCore::SimpleRange&)range;
- (std::optional<WebCore::SimpleRange>)_convertToDOMRange:(NSRange)range;
- (std::optional<WebCore::SimpleRange>)_convertToDOMRange:(NSRange)range rangeIsRelativeTo:(WebRangeIsRelativeTo)rangeIsRelativeTo;

- (DOMDocumentFragment *)_documentFragmentWithMarkupString:(NSString *)markupString baseURLString:(NSString *)baseURLString;
- (DOMDocumentFragment *)_documentFragmentWithNodesAsParagraphs:(NSArray *)nodes;

- (void)_replaceSelectionWithNode:(DOMNode *)node selectReplacement:(BOOL)selectReplacement smartReplace:(BOOL)smartReplace matchStyle:(BOOL)matchStyle;

- (void)_insertParagraphSeparatorInQuotedContent;

- (DOMRange *)_characterRangeAtPoint:(NSPoint)point;

- (DOMCSSStyleDeclaration *)_typingStyle;
- (void)_setTypingStyle:(DOMCSSStyleDeclaration *)style withUndoAction:(WebCore::EditAction)undoAction;

#if ENABLE(DRAG_SUPPORT) && PLATFORM(MAC)
- (void)_dragSourceEndedAt:(NSPoint)windowLoc operation:(NSDragOperation)operation;
#endif

- (BOOL)_canProvideDocumentSource;
- (BOOL)_canSaveAsWebArchive;

- (void)_commitData:(NSData *)data;

@end

@interface NSObject (WebInternalFrameLoadDelegate)
- (void)webFrame:(WebFrame *)webFrame didFinishLoadWithError:(NSError *)error;
@end
