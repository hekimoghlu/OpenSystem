/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#import "config.h"
#import "WKTextInputWindowController.h"

#if USE(APPKIT)

#import <Carbon/Carbon.h>
#import <pal/spi/mac/HIToolboxSPI.h>
#import <pal/system/mac/WebPanel.h>
#import <wtf/NeverDestroyed.h>
#import <wtf/RetainPtr.h>

@interface WKTextInputView : NSTextView
@end

@implementation WKTextInputView

- (NSArray *)validAttributesForMarkedText
{
    // Let TSM know that a bottom input window would be created for marked text.
    return [[super validAttributesForMarkedText] arrayByAddingObject:@"__NSUsesFloatingInputWindow"];
}

@end

@interface WKTextInputPanel : WebPanel {
    RetainPtr<NSTextView> _inputTextView;
}

- (NSTextInputContext *)_inputContext;
- (BOOL)_interpretKeyEvent:(NSEvent *)event usingLegacyCocoaTextInput:(BOOL)usingLegacyCocoaTextInput string:(NSString **)string;

- (BOOL)_hasMarkedText;
- (void)_unmarkText;

@end

#define inputWindowHeight 20

@implementation WKTextInputPanel

- (void)dealloc
{
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    
    [super dealloc];
}

- (id)init
{
    self = [super init];
    if (!self)
        return nil;
    
    // Set the frame size.
    NSRect visibleFrame = [[NSScreen mainScreen] visibleFrame];
    NSRect frame = NSMakeRect(visibleFrame.origin.x, visibleFrame.origin.y, visibleFrame.size.width, inputWindowHeight);
     
    [self setFrame:frame display:NO];
        
    _inputTextView = adoptNS([[WKTextInputView alloc] initWithFrame:[(NSView *)self.contentView frame]]);
    [_inputTextView setAutoresizingMask: NSViewWidthSizable | NSViewHeightSizable | NSViewMaxXMargin | NSViewMinXMargin | NSViewMaxYMargin | NSViewMinYMargin];
        
    auto scrollView = adoptNS([[NSScrollView alloc] initWithFrame:[(NSView *)self.contentView frame]]);
    [scrollView setDocumentView: _inputTextView.get()];
    self.contentView = scrollView.get();
        
    [self setFloatingPanel:YES];

    return self;
}

- (void)_unmarkText
{
    [_inputTextView setString:@""];
    [self orderOut:nil];
}

- (BOOL)_interpretKeyEvent:(NSEvent *)event usingLegacyCocoaTextInput:(BOOL)usingLegacyCocoaTextInput string:(NSString **)string
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    BOOL hadMarkedText = [_inputTextView hasMarkedText];
ALLOW_DEPRECATED_DECLARATIONS_END
 
    *string = nil;

    // Let TSM know that a bottom input window would be created for marked text.
    // FIXME: Can be removed once we can rely on __NSUsesFloatingInputWindow (or a better API) being available everywhere.
    EventRef carbonEvent = static_cast<EventRef>(const_cast<void*>([event eventRef]));
    if (carbonEvent) {
        Boolean ignorePAH = true;
        SetEventParameter(carbonEvent, 'iPAH', typeBoolean, sizeof(ignorePAH), &ignorePAH);
    }

    if (![[_inputTextView inputContext] handleEvent:event])
        return NO;
    
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    if ([_inputTextView hasMarkedText]) {
ALLOW_DEPRECATED_DECLARATIONS_END
        // Don't show the input method window for dead keys
        if ([[event characters] length] > 0)
            [self orderFront:nil];

        return YES;
    }

    bool shouldReturnTextString = hadMarkedText;

    // In the updated Cocoa text input model spec, we always want to return the text even if the text view didn't have marked text.
    if (!usingLegacyCocoaTextInput)
        shouldReturnTextString = true;

    if (shouldReturnTextString) {
        [[_inputTextView inputContext] discardMarkedText]; // Don't show Cangjie predictive candidates, cf. <rdar://14458297>.
        [self orderOut:nil];

        NSString *text = [[_inputTextView textStorage] string];
        if ([text length] > 0)
            *string = adoptNS([text copy]).autorelease();
    }
            
    [_inputTextView setString:@""];
    return hadMarkedText;
}

- (NSTextInputContext *)_inputContext
{
    return [_inputTextView inputContext];
}

- (BOOL)_hasMarkedText
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return [_inputTextView hasMarkedText];
ALLOW_DEPRECATED_DECLARATIONS_END
}

@end

@implementation WKTextInputWindowController

+ (WKTextInputWindowController *)sharedTextInputWindowController
{
    static NeverDestroyed<RetainPtr<WKTextInputWindowController>> textInputWindowController = adoptNS([[WKTextInputWindowController alloc] init]);
    return textInputWindowController.get().get();
}

- (id)init
{
    self = [super init];
    if (!self)
        return nil;
    
    _panel = [[WKTextInputPanel alloc] init];
    
    return self;
}

- (NSTextInputContext *)inputContext
{
    return [_panel _inputContext];
}

- (BOOL)hasMarkedText
{
    return [_panel _hasMarkedText];
}

- (BOOL)interpretKeyEvent:(NSEvent *)event usingLegacyCocoaTextInput:(BOOL)usingLegacyCocoaTextInput string:(NSString **)string
{
    return [_panel _interpretKeyEvent:event usingLegacyCocoaTextInput:usingLegacyCocoaTextInput string:string];
}

- (void)unmarkText
{
    [_panel _unmarkText];
}

@end

#endif // USE(APPKIT)
