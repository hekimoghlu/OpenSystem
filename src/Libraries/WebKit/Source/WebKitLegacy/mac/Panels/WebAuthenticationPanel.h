/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 24, 2024.
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

#import <Cocoa/Cocoa.h>
#import <Foundation/NSURLCredentialStorage.h>

@class NSURLAuthenticationChallenge;

@interface WebAuthenticationPanel : NSObject
{
    IBOutlet NSTextField *mainLabel;
    IBOutlet NSPanel *panel;
    IBOutlet NSTextField *password;
    IBOutlet NSTextField *smallLabel;
    IBOutlet NSTextField *username;
    IBOutlet NSImageView *imageView;
    IBOutlet NSButton *remember;
    IBOutlet NSTextField *separateRealmLabel;
    BOOL nibLoaded;
    BOOL usingSheet;
    id callback;
    SEL selector;
    NSURLAuthenticationChallenge *challenge;
}

-(id)initWithCallback:(id)cb selector:(SEL)sel;

// Interface-related methods
- (IBAction)cancel:(id)sender;
- (IBAction)logIn:(id)sender;

- (BOOL)loadNib;

- (void)runAsModalDialogWithChallenge:(NSURLAuthenticationChallenge *)chall;
- (void)runAsSheetOnWindow:(NSWindow *)window withChallenge:(NSURLAuthenticationChallenge *)chall;

- (void)sheetDidEnd:(NSWindow *)sheet returnCode:(int)returnCode contextInfo:(void  *)contextInfo;

@end

// This is in the header so it can be used from the nib file
@interface WebNonBlockingPanel : NSPanel
@end

#endif // !PLATFORM(IOS_FAMILY)
