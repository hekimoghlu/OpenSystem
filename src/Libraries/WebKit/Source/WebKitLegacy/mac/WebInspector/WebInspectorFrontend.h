/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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
#import <wtf/NakedPtr.h>

class WebInspectorFrontendClient;

@interface WebInspectorFrontend : NSObject {
@private
    NakedPtr<WebInspectorFrontendClient> m_frontendClient;
}
- (id)initWithFrontendClient:(NakedPtr<WebInspectorFrontendClient>)frontendClient;
- (void)attach;
- (void)detach;
- (void)close;

- (BOOL)isDebuggingEnabled;
- (void)setDebuggingEnabled:(BOOL)enabled;

- (BOOL)isTimelineProfilingEnabled;
- (void)setTimelineProfilingEnabled:(BOOL)enabled;

- (BOOL)isProfilingJavaScript;
- (void)startProfilingJavaScript;
- (void)stopProfilingJavaScript;

- (void)showConsole;

@end
