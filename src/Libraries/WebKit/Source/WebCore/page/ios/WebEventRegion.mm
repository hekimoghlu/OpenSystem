/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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
#import "WebEventRegion.h"
 
#if ENABLE(IOS_TOUCH_EVENTS)

#import "FloatQuad.h"

using namespace WebCore;

@interface WebEventRegion(Private)
- (FloatQuad)quad;
@end

@implementation WebEventRegion

- (id)initWithPoints:(CGPoint)inP1 :(CGPoint)inP2 :(CGPoint)inP3 :(CGPoint)inP4
{
    if (!(self = [super init]))
        return nil;
        
    p1 = inP1;
    p2 = inP2;
    p3 = inP3;
    p4 = inP4;
    return self;
}

- (id)copyWithZone:(NSZone *)zone
{
    UNUSED_PARAM(zone);
    return [self retain];
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"p1:{%g, %g} p2:{%g, %g} p3:{%g, %g} p4:{%g, %g}", p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y];
}

- (BOOL)hitTest:(CGPoint)point
{
    return [self quad].containsPoint(point);
}

// FIXME: Overriding isEqual: without overriding hash will cause trouble if this ever goes into an NSSet or is the key in an NSDictionary,
// since two equal objects could have different hashes.
- (BOOL)isEqual:(id)other
{
    if (![other isKindOfClass:[WebEventRegion class]])
        return NO;
    return CGPointEqualToPoint(p1, ((WebEventRegion *)other)->p1)
        && CGPointEqualToPoint(p2, ((WebEventRegion *)other)->p2)
        && CGPointEqualToPoint(p3, ((WebEventRegion *)other)->p3)
        && CGPointEqualToPoint(p4, ((WebEventRegion *)other)->p4);
}

- (FloatQuad)quad
{
    return FloatQuad(p1, p2, p3, p4);
}

- (CGPoint)p1
{
    return p1;
}

- (CGPoint)p2
{
    return p2;
}

- (CGPoint)p3
{
    return p3;
}

- (CGPoint)p4
{
    return p4;
}

@end

#endif // ENABLE(IOS_TOUCH_EVENTS)
