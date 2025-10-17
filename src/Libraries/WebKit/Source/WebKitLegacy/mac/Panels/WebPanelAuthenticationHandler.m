/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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

#import <WebKitLegacy/WebPanelAuthenticationHandler.h>

#import <Foundation/NSURLAuthenticationChallenge.h>
#import <WebKitLegacy/WebAuthenticationPanel.h>
#import <wtf/Assertions.h>

static NSString *WebModalDialogPretendWindow = @"WebModalDialogPretendWindow";

@implementation WebPanelAuthenticationHandler {
    NSMapTable *windowToPanel;
    NSMapTable *challengeToWindow;
    NSMapTable *windowToChallengeQueue;
}

WebPanelAuthenticationHandler *sharedHandler;

+ (id)sharedHandler
{
    if (sharedHandler == nil)
        sharedHandler = [[self alloc] init];
    return sharedHandler;
}

-(id)init
{
    self = [super init];
    if (self != nil) {
        windowToPanel = [[NSMapTable alloc] initWithKeyOptions:NSPointerFunctionsStrongMemory valueOptions:NSPointerFunctionsStrongMemory capacity:0];
        challengeToWindow = [[NSMapTable alloc] initWithKeyOptions:NSPointerFunctionsStrongMemory valueOptions:NSPointerFunctionsStrongMemory capacity:0];
        windowToChallengeQueue = [[NSMapTable alloc] initWithKeyOptions:NSPointerFunctionsStrongMemory valueOptions:NSPointerFunctionsStrongMemory capacity:0];
    }

    return self;
}

-(void)dealloc
{
    [windowToPanel release];
    [challengeToWindow release];    
    [windowToChallengeQueue release];    
    [super dealloc];
}

-(void)enqueueChallenge:(NSURLAuthenticationChallenge *)challenge forWindow:(id)window
{
    NSMutableArray *queue = [windowToChallengeQueue objectForKey:window];
    if (!queue) {
        queue = [[NSMutableArray alloc] init];
        [windowToChallengeQueue setObject:queue forKey:window];
        [queue release];
    }
    [queue addObject:challenge];
}

-(void)tryNextChallengeForWindow:(id)window
{
    NSMutableArray *queue = [windowToChallengeQueue objectForKey:window];
    if (!queue)
        return;

    NSURLAuthenticationChallenge *challenge = [[queue objectAtIndex:0] retain];
    [queue removeObjectAtIndex:0];
    if (![queue count])
        [windowToChallengeQueue removeObjectForKey:window];

    NSURLCredential *latestCredential = [[NSURLCredentialStorage sharedCredentialStorage] defaultCredentialForProtectionSpace:[challenge protectionSpace]];

    if ([latestCredential hasPassword]) {
        [[challenge sender] useCredential:latestCredential forAuthenticationChallenge:challenge];
        [challenge release];
        return;
    }
                                                                    
    [self startAuthentication:challenge window:(window == WebModalDialogPretendWindow ? nil : window)];
    [challenge release];
}


-(void)startAuthentication:(NSURLAuthenticationChallenge *)challenge window:(NSWindow *)w
{
    id window = w ? (id)w : (id)WebModalDialogPretendWindow;

    if ([windowToPanel objectForKey:window] != nil) {
        [self enqueueChallenge:challenge forWindow:window];
        return;
    }

    // In this case, we have an attached sheet that's not one of our
    // authentication panels, so enqueing is not an option. Just
    // cancel loading instead, since this case is fairly
    // unlikely (how would you be loading a page if you had an error
    // sheet up?)
    if ([w attachedSheet] != nil) {
        [[challenge sender] cancelAuthenticationChallenge:challenge];
        return;
    }

    WebAuthenticationPanel *panel = [[WebAuthenticationPanel alloc] initWithCallback:self selector:@selector(_authenticationDoneWithChallenge:result:)];
    [challengeToWindow setObject:window forKey:challenge];
    [windowToPanel setObject:panel forKey:window];
    [panel release];
    
    if (window == WebModalDialogPretendWindow)
        [panel runAsModalDialogWithChallenge:challenge];
    else
        [panel runAsSheetOnWindow:window withChallenge:challenge];
}

-(void)cancelAuthentication:(NSURLAuthenticationChallenge *)challenge
{
    id window = [challengeToWindow objectForKey:challenge];
    if (!window)
        return;

    WebAuthenticationPanel *panel = [windowToPanel objectForKey:window];
    [panel cancel:self];
}

-(void)_authenticationDoneWithChallenge:(NSURLAuthenticationChallenge *)challenge result:(NSURLCredential *)credential
{
    id window = [challengeToWindow objectForKey:challenge];
    [window retain];
    if (window != nil) {
        [windowToPanel removeObjectForKey:window];
        [challengeToWindow removeObjectForKey:challenge];
    }

    if (credential == nil) {
        [[challenge sender] continueWithoutCredentialForAuthenticationChallenge:challenge];
    } else {
        [[challenge sender] useCredential:credential forAuthenticationChallenge:challenge];
    }

    [self tryNextChallengeForWindow:window];
    [window release];
}

@end

#endif // !PLATFORM(IOS_FAMILY)
