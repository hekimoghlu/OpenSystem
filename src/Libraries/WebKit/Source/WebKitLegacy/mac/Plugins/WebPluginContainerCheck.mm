/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 31, 2024.
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
#import "WebPluginContainerCheck.h"

#import "WebFrameInternal.h"
#import "WebPluginContainerPrivate.h"
#import "WebPluginController.h"
#import "WebPolicyDelegatePrivate.h"
#import "WebView.h"
#import "WebViewInternal.h"
#import <Foundation/NSDictionary.h>
#import <Foundation/NSURL.h>
#import <Foundation/NSURLRequest.h>
#import <WebCore/Document.h>
#import <WebCore/FrameLoader.h>
#import <WebCore/FrameLoaderTypes.h>
#import <WebCore/LocalFrame.h>
#import <WebCore/OriginAccessPatterns.h>
#import <WebCore/SecurityOrigin.h>
#import <wtf/Assertions.h>
#import <wtf/ObjCRuntimeExtras.h>

#if PLATFORM(IOS_FAMILY)
@interface WebPluginController (SecretsIKnow)
- (WebFrame *)webFrame; // FIXME: This file calls -[WebPluginController webFrame], which is not declared in WebPluginController.h.  Merge issue?  Are the plug-in files out of date?
@end
#endif

@implementation WebPluginContainerCheck

- (id)initWithRequest:(NSURLRequest *)request target:(NSString *)target resultObject:(id)obj selector:(SEL)selector controller:(id <WebPluginContainerCheckController>)controller contextInfo:(id)contextInfo /*optional*/
{
    if (!(self = [super init]))
        return nil;
    
    _request = [request copy];
    _target = [target copy];
    _resultObject = [obj retain];
    _resultSelector = selector;
    _contextInfo = [contextInfo retain];
    
    // controller owns us so don't retain, to avoid cycle
    _controller = controller;
    
    return self;
}

+ (id)checkWithRequest:(NSURLRequest *)request target:(NSString *)target resultObject:(id)obj selector:(SEL)selector controller:(id <WebPluginContainerCheckController>)controller contextInfo:(id)contextInfo /*optional*/
{
    return adoptNS([[self alloc] initWithRequest:request target:target resultObject:obj selector:selector controller:controller contextInfo:contextInfo]).autorelease();
}

- (void)dealloc
{
    // mandatory to complete or cancel before releasing this object
    ASSERT(_done);
    [super dealloc];
}

- (void)_continueWithPolicy:(WebCore::PolicyAction)policy
{
    if (_contextInfo)
        wtfObjCMsgSend<void>(_resultObject, _resultSelector, (policy == WebCore::PolicyAction::Use), _contextInfo);
    else     
        wtfObjCMsgSend<void>(_resultObject, _resultSelector, (policy == WebCore::PolicyAction::Use));

    // this will call indirectly call cancel
    [_controller _webPluginContainerCancelCheckIfAllowedToLoadRequest:self];
}

- (BOOL)_isForbiddenFileLoad
{
    auto* coreFrame = core([_controller webFrame]);
    ASSERT(coreFrame);
    if (!coreFrame->document()->securityOrigin().canDisplay([_request URL], WebCore::OriginAccessPatternsForWebProcess::singleton())) {
        [self _continueWithPolicy:WebCore::PolicyAction::Ignore];
        return YES;
    }

    return NO;
}

- (NSDictionary *)_actionInformationWithURL:(NSURL *)URL
{
    return @{
        WebActionNavigationTypeKey: @(WebNavigationTypePlugInRequest),
        WebActionModifierFlagsKey: @(0),
        WebActionOriginalURLKey: URL,
    };
}

- (void)_askPolicyDelegate
{
    WebView *webView = [_controller webView];

    WebFrame *targetFrame;
    if ([_target length] > 0) {
        targetFrame = [[_controller webFrame] findFrameNamed:_target];
    } else {
        targetFrame = [_controller webFrame];
    }

    NSDictionary *action = [self _actionInformationWithURL:[_request URL]];

    _listener = [[WebPolicyDecisionListener alloc] _initWithTarget:self action:@selector(_continueWithPolicy:)];

    if (targetFrame == nil) {
        // would open new window
        [[webView _policyDelegateForwarder] webView:webView
                     decidePolicyForNewWindowAction:action
                                            request:_request
                                       newFrameName:_target
                                   decisionListener:_listener];
    } else {
        // would target existing frame
        [[webView _policyDelegateForwarder] webView:webView
                    decidePolicyForNavigationAction:action
                                            request:_request
                                              frame:targetFrame
                                   decisionListener:_listener];        
    }
}

- (void)start
{
    ASSERT(!_listener);
    ASSERT(!_done);

    if ([self _isForbiddenFileLoad])
        return;

    [self _askPolicyDelegate];
}

- (void)cancel
{
    if (_done)
        return;

    [_request release];
    _request = nil;
    
    [_target release];
    _target = nil;

    [_listener _invalidate];
    [_listener release];
    _listener = nil;

    [_resultObject autorelease];
    _resultObject = nil;

    _controller = nil;
    
    [_contextInfo release];
    _contextInfo = nil;

    _done = YES;
}

- (id)contextInfo
{
    return _contextInfo;   
}

@end
