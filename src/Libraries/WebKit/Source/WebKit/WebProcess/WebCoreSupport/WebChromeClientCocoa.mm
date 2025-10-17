/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#import "WebChromeClient.h"

#if PLATFORM(COCOA)

#import "WebIconUtilities.h"
#import "WebPage.h"
#import <WebCore/AXObjectCache.h>
#import <WebCore/Icon.h>

#if PLATFORM(MAC)

#import "ApplicationServicesSPI.h"

extern "C" AXError _AXUIElementNotifyProcessSuspendStatus(AXSuspendStatus);

#endif // PLATFORM(MAC)

namespace WebKit {
using namespace WebCore;

RefPtr<Icon> WebChromeClient::createIconForFiles(const Vector<String>& filenames)
{
    return Icon::create(iconForFiles(filenames).get());
}

void AXRelayProcessSuspendedNotification::sendProcessSuspendMessage(bool suspended)
{
    if (!AXObjectCache::accessibilityEnabled())
        return;

#if PLATFORM(MAC)
    _AXUIElementNotifyProcessSuspendStatus(suspended ? AXSuspendStatusSuspended : AXSuspendStatusRunning);
#else
    NSDictionary *message = @{ @"pid" : @(getpid()), @"suspended" : @(suspended) };
    NSData *data = [NSKeyedArchiver archivedDataWithRootObject:message requiringSecureCoding:YES error:nil];
    m_page->relayAccessibilityNotification("AXProcessSuspended"_s, data);
#endif
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
