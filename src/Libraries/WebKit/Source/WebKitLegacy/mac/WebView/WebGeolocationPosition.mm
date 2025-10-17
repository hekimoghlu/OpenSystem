/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#import "WebGeolocationPosition.h"

#import "WebGeolocationPositionInternal.h"
#import <WebCore/GeolocationPositionData.h>
#import <wtf/RefPtr.h>

using namespace WebCore;

@interface WebGeolocationPositionInternal : NSObject
{
@public
    GeolocationPositionData _position;
}

- (id)initWithCoreGeolocationPosition:(GeolocationPositionData&&)coreGeolocationPosition;
@end

@implementation WebGeolocationPositionInternal

- (id)initWithCoreGeolocationPosition:(GeolocationPositionData&&)coreGeolocationPosition
{
    self = [super init];
    if (!self)
        return nil;
    _position = WTFMove(coreGeolocationPosition);
    return self;
}

@end

@implementation WebGeolocationPosition

std::optional<GeolocationPositionData> core(WebGeolocationPosition *position)
{
    if (!position)
        return std::nullopt;
    return position->_internal->_position;
}

- (id)initWithTimestamp:(double)timestamp latitude:(double)latitude longitude:(double)longitude accuracy:(double)accuracy
{
    self = [super init];
    if (!self)
        return nil;
    _internal = [[WebGeolocationPositionInternal alloc] initWithCoreGeolocationPosition:GeolocationPositionData { timestamp, latitude, longitude, accuracy }];
    return self;
}

- (id)initWithGeolocationPosition:(GeolocationPositionData&&)coreGeolocationPosition
{
    self = [super init];
    if (!self)
        return nil;
    _internal = [[WebGeolocationPositionInternal alloc] initWithCoreGeolocationPosition:WTFMove(coreGeolocationPosition)];
    return self;
}

- (void)dealloc
{
    [_internal release];
    [super dealloc];
}

@end
