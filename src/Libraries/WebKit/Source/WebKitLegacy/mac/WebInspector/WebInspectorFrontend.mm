/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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
#import "WebInspectorFrontend.h"

#import "WebInspectorClient.h"
#import <WebCore/InspectorFrontendClient.h>

using namespace WebCore;

@implementation WebInspectorFrontend

- (id)initWithFrontendClient:(NakedPtr<WebInspectorFrontendClient>)frontendClient
{
    if (!(self = [super init]))
        return nil;

    m_frontendClient = frontendClient;
    return self;
}

- (void)attach
{
    m_frontendClient->attachWindow(InspectorFrontendClient::DockSide::Bottom);
}

- (void)detach
{
    m_frontendClient->detachWindow();
}

- (void)close
{
    m_frontendClient->closeWindow();
}

- (BOOL)isDebuggingEnabled
{
    return m_frontendClient->isDebuggingEnabled();
}

- (void)setDebuggingEnabled:(BOOL)enabled
{
    m_frontendClient->setDebuggingEnabled(enabled);
}

- (BOOL)isProfilingJavaScript
{
    return m_frontendClient->isProfilingJavaScript();
}

- (void)startProfilingJavaScript
{
    m_frontendClient->startProfilingJavaScript();
}

- (void)stopProfilingJavaScript
{
    m_frontendClient->stopProfilingJavaScript();
}

- (BOOL)isTimelineProfilingEnabled
{
    return m_frontendClient->isTimelineProfilingEnabled();
}

- (void)setTimelineProfilingEnabled:(BOOL)enabled
{
    m_frontendClient->setTimelineProfilingEnabled(enabled);
}

- (void)showConsole
{
    m_frontendClient->showConsole();
}

@end
