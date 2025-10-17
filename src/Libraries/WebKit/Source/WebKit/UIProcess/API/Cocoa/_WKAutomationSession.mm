/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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
#import "_WKAutomationSessionInternal.h"

#import "AutomationSessionClient.h"
#import "WKAPICast.h"
#import "WKProcessPool.h"
#import "WebAutomationSession.h"
#import "_WKAutomationSessionConfiguration.h"
#import "_WKAutomationSessionDelegate.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/WeakObjCPtr.h>

@implementation _WKAutomationSession {
    RetainPtr<_WKAutomationSessionConfiguration> _configuration;
    WeakObjCPtr<id <_WKAutomationSessionDelegate>> _delegate;
}

- (instancetype)init
{
    return [self initWithConfiguration:adoptNS([[_WKAutomationSessionConfiguration alloc] init]).get()];
}

- (instancetype)initWithConfiguration:(_WKAutomationSessionConfiguration *)configuration
{
    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<WebKit::WebAutomationSession>(self);

    _configuration = adoptNS([configuration copy]);

    return self;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKAutomationSession.class, self))
        return;

    _session->setClient(nullptr);
    _session->~WebAutomationSession();

    [super dealloc];
}

- (id <_WKAutomationSessionDelegate>)delegate
{
    return _delegate.getAutoreleased();
}

- (void)setDelegate:(id <_WKAutomationSessionDelegate>)delegate
{
    _delegate = delegate;
    _session->setClient(delegate ? makeUnique<WebKit::AutomationSessionClient>(delegate) : nullptr);
}

- (NSString *)sessionIdentifier
{
    return _session->sessionIdentifier();
}

- (void)setSessionIdentifier:(NSString *)sessionIdentifier
{
    _session->setSessionIdentifier(sessionIdentifier);
}

- (_WKAutomationSessionConfiguration *)configuration
{
    return adoptNS([_configuration copy]).autorelease();
}

- (BOOL)isPaired
{
    return _session->isPaired();
}

- (BOOL)isPendingTermination
{
    return _session->isPendingTermination();
}

- (BOOL)isSimulatingUserInteraction
{
    return _session->isSimulatingUserInteraction();
}

- (void)terminate
{
    _session->terminate();
}

#if PLATFORM(MAC)
- (BOOL)wasEventSynthesizedForAutomation:(NSEvent *)event
{
    return _session->wasEventSynthesizedForAutomation(event);
}

- (void)markEventAsSynthesizedForAutomation:(NSEvent *)event
{
    _session->markEventAsSynthesizedForAutomation(event);
}
#endif

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_session;
}

@end
