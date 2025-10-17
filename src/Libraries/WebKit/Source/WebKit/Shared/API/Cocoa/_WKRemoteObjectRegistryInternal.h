/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#import "_WKRemoteObjectRegistry.h"
#import <wtf/NakedRef.h>

namespace IPC {
class MessageSender;
}

namespace WebKit {
class RemoteObjectInvocation;
class RemoteObjectRegistry;
class UserData;
class WebPage;
class WebPageProxy;
}

@interface _WKRemoteObjectRegistry ()

@property (nonatomic, readonly) WebKit::RemoteObjectRegistry& remoteObjectRegistry;

- (id)_initWithWebPage:(NakedRef<WebKit::WebPage>)messageSender;
- (id)_initWithWebPageProxy:(NakedRef<WebKit::WebPageProxy>)messageSender;
- (void)_invalidate;

- (void)_sendInvocation:(NSInvocation *)invocation interface:(_WKRemoteObjectInterface *)interface;
- (void)_invokeMethod:(const WebKit::RemoteObjectInvocation&)invocation;

- (void)_callReplyWithID:(uint64_t)replyID blockInvocation:(const WebKit::UserData&)blockInvocation;
- (void)_releaseReplyWithID:(uint64_t)replyID;

@end
