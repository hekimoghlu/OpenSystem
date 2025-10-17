/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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
#import "WebDeviceOrientation.h"

#import "WebDeviceOrientationProvider.h"
#import <WebCore/DeviceOrientationClientMock.h>
#import <WebCore/DeviceOrientationData.h>
#import <wtf/RefPtr.h>

@interface WebDeviceOrientationInternal : NSObject {
@public
    RefPtr<WebCore::DeviceOrientationData> m_orientation;
}

- (id)initWithCoreDeviceOrientation:(RefPtr<WebCore::DeviceOrientationData>&&)coreDeviceOrientation;
@end

@interface WebDeviceOrientation (Internal)

- (id)initWithCoreDeviceOrientation:(RefPtr<WebCore::DeviceOrientationData>&&)coreDeviceOrientation;

@end

WebCore::DeviceOrientationData* core(WebDeviceOrientation*);

@protocol WebDeviceOrientationProviderMock <WebDeviceOrientationProvider>
- (void)setController:(WebCore::DeviceOrientationController*)controller;
@end
