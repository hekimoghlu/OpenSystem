/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#import "_WKPageLoadTimingInternal.h"

#import "WebPageLoadTiming.h"
#import <WebCore/WebCoreObjCExtras.h>

static NSDate *nsDateFromMonotonicTime(WallTime time)
{
    if (!time)
        return nil;
    return [NSDate dateWithTimeIntervalSince1970:time.secondsSinceEpoch().value()];
}

@implementation _WKPageLoadTiming {
    WallTime _navigationStart;
    WallTime _firstVisualLayout;
    WallTime _firstMeaningfulPaint;
    WallTime _documentFinishedLoading;
    WallTime _allSubresourcesFinishedLoading;
}

- (instancetype)_initWithTiming:(const WebKit::WebPageLoadTiming&)timing
{
    if (!(self = [super init]))
        return nil;

    _navigationStart = timing.navigationStart();
    _firstVisualLayout = timing.firstVisualLayout();
    _firstMeaningfulPaint = timing.firstMeaningfulPaint();
    _documentFinishedLoading = timing.documentFinishedLoading();
    _allSubresourcesFinishedLoading = timing.allSubresourcesFinishedLoading();

    return self;
}

- (NSDate *)navigationStart
{
    return nsDateFromMonotonicTime(_navigationStart);
}

- (NSDate *)firstVisualLayout
{
    return nsDateFromMonotonicTime(_firstVisualLayout);
}

- (NSDate *)firstMeaningfulPaint
{
    return nsDateFromMonotonicTime(_firstMeaningfulPaint);
}

- (NSDate *)documentFinishedLoading
{
    return nsDateFromMonotonicTime(_documentFinishedLoading);
}

- (NSDate *)allSubresourcesFinishedLoading
{
    return nsDateFromMonotonicTime(_allSubresourcesFinishedLoading);
}

@end
