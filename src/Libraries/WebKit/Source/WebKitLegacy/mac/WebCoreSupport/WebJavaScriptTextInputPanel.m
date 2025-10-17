/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
#if !PLATFORM(IOS_FAMILY)

#import "WebJavaScriptTextInputPanel.h"

#import <wtf/Assertions.h>

#import <WebKitLegacy/WebNSControlExtras.h>
#import <WebKitLegacy/WebNSWindowExtras.h>

@implementation WebJavaScriptTextInputPanel {
    IBOutlet NSTextField *prompt;
    IBOutlet NSTextField *textInput;
}

- (id)initWithPrompt:(NSString *)p text:(NSString *)t
{
    self = [self initWithWindowNibName:@"WebJavaScriptTextInputPanel"];
    if (!self)
        return nil;
    NSWindow *window = [self window];
    
    // This must be done after the call to [self window], because
    // until then, prompt and textInput will be nil.
    ASSERT(prompt);
    ASSERT(textInput);
    [prompt setStringValue:p];
    [textInput setStringValue:t];

    [prompt sizeToFitAndAdjustWindowHeight];
    [window centerOverMainWindow];
    
    return self;
}

- (NSString *)text
{
    return [textInput stringValue];
}

- (IBAction)pressedCancel:(id)sender
{
    [NSApp stopModalWithCode:NO];
}

- (IBAction)pressedOK:(id)sender
{
    [NSApp stopModalWithCode:YES];
}

@end

#endif // !PLATFORM(IOS_FAMILY)
