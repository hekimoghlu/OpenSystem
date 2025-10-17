/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
#import "WebStorageTrackerClient.h"

#import "WebSecurityOriginInternal.h"
#import "WebStorageManagerPrivate.h"
#import <WebCore/SecurityOrigin.h>
#import <WebCore/SecurityOriginData.h>
#import <wtf/MainThread.h>
#import <wtf/RetainPtr.h>
#import <wtf/text/WTFString.h>

using namespace WebCore;

WebStorageTrackerClient* WebStorageTrackerClient::sharedWebStorageTrackerClient()
{
    static WebStorageTrackerClient* sharedClient = new WebStorageTrackerClient();
    return sharedClient;
}

WebStorageTrackerClient::WebStorageTrackerClient()
{
}

WebStorageTrackerClient::~WebStorageTrackerClient()
{
}

void WebStorageTrackerClient::dispatchDidModifyOrigin(SecurityOrigin* origin)
{
    RetainPtr<WebSecurityOrigin> webSecurityOrigin = adoptNS([[WebSecurityOrigin alloc] _initWithWebCoreSecurityOrigin:origin]);

    [[NSNotificationCenter defaultCenter] postNotificationName:WebStorageDidModifyOriginNotification 
                                                        object:webSecurityOrigin.get()];
}

void WebStorageTrackerClient::dispatchDidModifyOrigin(const String& originIdentifier)
{
    auto origin = SecurityOriginData::fromDatabaseIdentifier(originIdentifier);

    if (!origin) {
        ASSERT_NOT_REACHED();
        return;
    }
    
    if (isMainThread()) {
        dispatchDidModifyOrigin(origin->securityOrigin().ptr());
        return;
    }

    callOnMainThread([origin = origin->securityOrigin()->isolatedCopy()]() mutable {
        WebStorageTrackerClient::sharedWebStorageTrackerClient()->dispatchDidModifyOrigin(origin.ptr());
    });
}

void WebStorageTrackerClient::didFinishLoadingOrigins()
{
}
