/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
#import "WebDeviceOrientationProviderMockInternal.h"

#import "WebDeviceOrientationInternal.h"
#import <wtf/RetainPtr.h>

using namespace WebCore;

@implementation WebDeviceOrientationProviderMockInternal

- (id)init
{
    self = [super init];
    if (!self)
        return nil;
    m_core = makeUnique<DeviceOrientationClientMock>();
    return self;
}

- (void)setOrientation:(WebDeviceOrientation*)orientation
{
    m_core->setOrientation(core(orientation));
}

- (void)setController:(DeviceOrientationController*)controller
{
    m_core->setController(controller);
}

- (void)startUpdating
{
    m_core->startUpdating();
}

- (void)stopUpdating
{
    m_core->stopUpdating();
}

- (WebDeviceOrientation*)lastOrientation
{
    return adoptNS([[WebDeviceOrientation alloc] initWithCoreDeviceOrientation:m_core->lastOrientation()]).autorelease();
}

@end

@implementation WebDeviceOrientationProviderMock

+ (WebDeviceOrientationProviderMock *)shared
{
    static WebDeviceOrientationProviderMock *provider = [[WebDeviceOrientationProviderMock alloc] init];
    return provider;
}

- (id)init
{
    self = [super init];
    if (!self)
        return nil;
    m_internal = [[WebDeviceOrientationProviderMockInternal alloc] init];
    return self;
}

- (void)dealloc
{
    [m_internal release];
    [super dealloc];
}

- (void)setOrientation:(WebDeviceOrientation*)orientation
{
    [m_internal setOrientation:orientation];
}

- (void)startUpdating
{
    [m_internal startUpdating];
}

- (void)stopUpdating
{
    [m_internal stopUpdating];
}

- (WebDeviceOrientation*)lastOrientation
{
    return [m_internal lastOrientation];
}

- (void)setController:(WebCore::DeviceOrientationController*)controller
{
   [m_internal setController:controller];
}

@end
