/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
#import "_WKResourceLoadStatisticsThirdPartyInternal.h"

#import "APIArray.h"
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/cocoa/VectorCocoa.h>

@implementation _WKResourceLoadStatisticsThirdParty

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKResourceLoadStatisticsThirdParty.class, self))
        return;
    _thirdParty->API::ResourceLoadStatisticsThirdParty::~ResourceLoadStatisticsThirdParty();
    [super dealloc];
}

- (NSString *)thirdPartyDomain
{
    return _thirdParty->thirdPartyDomain();
}

- (NSArray<_WKResourceLoadStatisticsFirstParty *> *)underFirstParties
{
    return createNSArray(_thirdParty->underFirstParties(), [] (auto& data) {
        return wrapper(API::ResourceLoadStatisticsFirstParty::create(data));
    }).autorelease();
}

- (API::Object&)_apiObject
{
    return *_thirdParty;
}


@end

