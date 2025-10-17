/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 4, 2024.
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
#import "WebDeviceOrientationInternal.h"

using namespace WebCore;

@implementation WebDeviceOrientationInternal

- (id)initWithCoreDeviceOrientation:(RefPtr<DeviceOrientationData>&&)coreDeviceOrientation
{
    self = [super init];
    if (!self)
        return nil;
    m_orientation = WTFMove(coreDeviceOrientation);
    return self;
}

@end

@implementation WebDeviceOrientation (Internal)

- (id)initWithCoreDeviceOrientation:(RefPtr<WebCore::DeviceOrientationData>&&)coreDeviceOrientation
{
    self = [super init];
    if (!self)
        return nil;
    m_internal = [[WebDeviceOrientationInternal alloc] initWithCoreDeviceOrientation:WTFMove(coreDeviceOrientation)];
    return self;
}

@end

@implementation WebDeviceOrientation

DeviceOrientationData* core(WebDeviceOrientation* orientation)
{
    return orientation ? orientation->m_internal->m_orientation.get() : 0;
}

static std::optional<double> convert(bool canProvide, double value)
{
    if (!canProvide)
        return std::nullopt;
    return value;
}

- (id)initWithCanProvideAlpha:(bool)canProvideAlpha alpha:(double)alpha canProvideBeta:(bool)canProvideBeta beta:(double)beta canProvideGamma:(bool)canProvideGamma gamma:(double)gamma
{
    self = [super init];
    if (!self)
        return nil;
#if PLATFORM(IOS_FAMILY)
    // We don't use this API, but make sure that it compiles with the new
    // compass parameters.
    m_internal = [[WebDeviceOrientationInternal alloc] initWithCoreDeviceOrientation:DeviceOrientationData::create(convert(canProvideAlpha, alpha), convert(canProvideBeta, beta), convert(canProvideGamma, gamma), std::nullopt, std::nullopt)];
#else
    m_internal = [[WebDeviceOrientationInternal alloc] initWithCoreDeviceOrientation:DeviceOrientationData::create(convert(canProvideAlpha, alpha), convert(canProvideBeta, beta), convert(canProvideGamma, gamma), std::nullopt)];
#endif
    return self;
}

- (void)dealloc
{
    [m_internal release];
    [super dealloc];
}

@end
