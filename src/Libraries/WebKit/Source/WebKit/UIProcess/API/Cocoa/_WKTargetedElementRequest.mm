/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 31, 2025.
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
#import "_WKTargetedElementRequestInternal.h"

#import <WebCore/WebCoreObjCExtras.h>

@implementation _WKTargetedElementRequest {
    RetainPtr<NSString> _searchText;
}

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKTargetedElementRequest.class, self))
        return;
    _request->API::TargetedElementRequest::~TargetedElementRequest();
    [super dealloc];
}

- (API::Object&)_apiObject
{
    return *_request;
}

- (instancetype)init
{
    if (!(self = [super init]))
        return nil;

    API::Object::constructInWrapper<API::TargetedElementRequest>(self);
    return self;
}

- (instancetype)initWithSearchText:(NSString *)searchText
{
    if (!(self = [self init]))
        return nil;

    _request->setSearchText(searchText);
    return self;
}

- (instancetype)initWithPoint:(CGPoint)point
{
    if (!(self = [self init]))
        return nil;

    _request->setPoint(point);
    return self;
}

- (instancetype)initWithSelectors:(NSArray<NSSet<NSString *> *> *)nsSelectorsForElement
{
    if (!(self = [self init]))
        return nil;

    WebCore::TargetedElementSelectors selectorsForElement;
    selectorsForElement.reserveInitialCapacity(nsSelectorsForElement.count);
    for (NSSet<NSString *> *nsSelectors in nsSelectorsForElement) {
        HashSet<String> selectors;
        selectors.reserveInitialCapacity(nsSelectors.count);
        for (NSString *selector in nsSelectors)
            selectors.add(selector);
        selectorsForElement.append(WTFMove(selectors));
    }

    _request->setSelectors(WTFMove(selectorsForElement));
    return self;
}

- (BOOL)canIncludeNearbyElements
{
    return _request->canIncludeNearbyElements();
}

- (void)setCanIncludeNearbyElements:(BOOL)value
{
    _request->setCanIncludeNearbyElements(value);
}

- (BOOL)shouldIgnorePointerEventsNone
{
    return _request->shouldIgnorePointerEventsNone();
}

- (void)setShouldIgnorePointerEventsNone:(BOOL)value
{
    _request->setShouldIgnorePointerEventsNone(value);
}

@end
