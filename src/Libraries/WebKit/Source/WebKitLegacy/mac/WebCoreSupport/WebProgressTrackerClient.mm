/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#import "WebProgressTrackerClient.h"

#import "WebFramePrivate.h"
#import "WebViewInternal.h"

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WebCoreThreadMessage.h>
#endif

using namespace WebCore;

WebProgressTrackerClient::WebProgressTrackerClient(WebView *webView)
    : m_webView(webView)
{
}

#if !PLATFORM(IOS_FAMILY)
void WebProgressTrackerClient::willChangeEstimatedProgress()
{
    [m_webView _willChangeValueForKey:_WebEstimatedProgressKey];
}

void WebProgressTrackerClient::didChangeEstimatedProgress()
{
    [m_webView _didChangeValueForKey:_WebEstimatedProgressKey];
}
#endif

void WebProgressTrackerClient::progressStarted(WebCore::LocalFrame& originatingProgressFrame)
{
#if !PLATFORM(IOS_FAMILY)
    [[NSNotificationCenter defaultCenter] postNotificationName:WebViewProgressStartedNotification object:m_webView];
#else
    WebThreadPostNotification(WebViewProgressStartedNotification, m_webView, nil);
#endif
}

void WebProgressTrackerClient::progressEstimateChanged(WebCore::LocalFrame&)
{
#if !PLATFORM(IOS_FAMILY)
    [[NSNotificationCenter defaultCenter] postNotificationName:WebViewProgressEstimateChangedNotification object:m_webView];
#else
    NSNumber *progress = [NSNumber numberWithFloat:[m_webView estimatedProgress]];
    CGColorRef bodyBackgroundColor = [[m_webView mainFrame] _bodyBackgroundColor];
    
    // Use a CFDictionary so we can add the CGColorRef without compile errors. And then thanks to
    // toll-free bridging we can pass the CFDictionary as an NSDictionary to postNotification.
    auto userInfo = adoptCF(CFDictionaryCreateMutable(kCFAllocatorDefault, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
    CFDictionaryAddValue(userInfo.get(), WebViewProgressEstimatedProgressKey, progress);
    if (bodyBackgroundColor)
        CFDictionaryAddValue(userInfo.get(), WebViewProgressBackgroundColorKey, bodyBackgroundColor);
    
    WebThreadPostNotification(WebViewProgressEstimateChangedNotification, m_webView, (NSDictionary *)userInfo.get());
#endif
}

void WebProgressTrackerClient::progressFinished(LocalFrame&)
{
#if !PLATFORM(IOS_FAMILY)
    [[NSNotificationCenter defaultCenter] postNotificationName:WebViewProgressFinishedNotification object:m_webView];
#endif
}
