/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
// Things internal to the WebKit framework; not SPI.

#import "WebHTMLViewPrivate.h"
#import <wtf/NakedPtr.h>

@class CALayer;
@class WebFrame;
@class WebPluginController;

namespace WebCore {
class CachedImage;
    class KeyboardEvent;
}

#if PLATFORM(MAC)

@interface WebHTMLView () <NSDraggingSource>
@end

#endif

@interface WebHTMLView (WebInternal)
- (void)_selectionChanged;

#if PLATFORM(MAC)
- (void)_updateFontPanel;
- (void)_setSoftSpaceRange:(NSRange)range;
#endif

- (BOOL)_canSmartCopyOrDelete;

- (WebFrame *)_frame;
- (void)closeIfNotCurrentView;

#if PLATFORM(MAC)
- (void)_lookUpInDictionaryFromMenu:(id)sender;
- (BOOL)_interpretKeyEvent:(NakedPtr<WebCore::KeyboardEvent>)event savingCommands:(BOOL)savingCommands;
- (DOMDocumentFragment *)_documentFragmentFromPasteboard:(NSPasteboard *)pasteboard;
- (NSEvent *)_mouseDownEvent;
- (BOOL)isGrammarCheckingEnabled;
- (void)setGrammarCheckingEnabled:(BOOL)flag;
- (void)toggleGrammarChecking:(id)sender;
- (void)setPromisedDragTIFFDataSource:(NakedPtr<WebCore::CachedImage>)source;
#endif

#if PLATFORM(IOS_FAMILY)
- (BOOL)_handleEditingKeyEvent:(WebCore::KeyboardEvent *)event;
#endif

- (void)_web_updateLayoutAndStyleIfNeededRecursive;
- (void)_destroyAllWebPlugins;
- (BOOL)_needsLayout;

#if PLATFORM(MAC)
- (void)attachRootLayer:(CALayer *)layer;
- (void)detachRootLayer;

- (BOOL)_web_isDrawingIntoLayer;
- (BOOL)_web_isDrawingIntoAcceleratedLayer;
#endif

#if PLATFORM(IOS_FAMILY)
- (void)_layoutIfNeeded;
#endif

#if PLATFORM(MAC)
- (void)_changeSpellingToWord:(NSString *)newWord;
- (void)_startAutoscrollTimer:(NSEvent *)event;
#endif

- (void)_stopAutoscrollTimer;

- (WebPluginController *)_pluginController;

- (void)_executeSavedKeypressCommands;

- (WebCore::ScrollbarWidth)_scrollbarWidthStyle;

@end

@interface WebHTMLView (RemovedAppKitSuperclassMethods)
#if PLATFORM(IOS_FAMILY)
- (void)delete:(id)sender;
- (void)transpose:(id)sender;
#endif
- (BOOL)hasMarkedText;
@end
