/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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
#import "RemoteLayerTreeLayers.h"

#if PLATFORM(COCOA)

#import "Logging.h"
#import "RemoteLayerTreeNode.h"
#import <WebCore/DynamicContentScalingDisplayList.h>
#import <WebCore/DynamicContentScalingTypes.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/MachSendRight.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
#import <CoreRE/RECGCommandsContext.h>
#endif

@implementation WKCompositingLayer {
#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    RetainPtr<CFDataRef> _displayListDataForTesting;
#endif
}

- (NSString *)description
{
    return WebKit::RemoteLayerTreeNode::appendLayerDescription(super.description, self);
}

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)

- (void)_setWKContents:(id)contents withDisplayList:(WebCore::DynamicContentScalingDisplayList&&)displayList replayForTesting:(BOOL)replay
{
    auto data = displayList.displayList()->createCFData();

    if (replay) {
        _displayListDataForTesting = data;
        [self setNeedsDisplay];
        return;
    }

    self.contents = contents;

    auto surfaces = displayList.takeSurfaces();
    auto ports = adoptNS([[NSMutableArray alloc] initWithCapacity:surfaces.size()]);
    for (MachSendRight& surface : surfaces) {
        // We `leakSendRight` because CAMachPortCreate "adopts" the incoming reference.
        RetainPtr portWrapper = adoptCF(CAMachPortCreate(surface.leakSendRight()));
        [ports addObject:static_cast<id>(portWrapper.get())];
    }

    [self setValue:bridge_cast(data.get()) forKeyPath:WKDynamicContentScalingContentsKey];
    [self setValue:ports.get() forKeyPath:WKDynamicContentScalingPortsKey];
    [self setNeedsDisplay];
}

- (void)drawInContext:(CGContextRef)context
{
    if (!_displayListDataForTesting)
        return;
    CGContextScaleCTM(context, 1, -1);
    CGContextTranslateCTM(context, 0, -self.bounds.size.height);
    RECGContextDrawCGCommandsEncodedData(context, _displayListDataForTesting.get(), nullptr);
}

#endif // ENABLE(RE_DYNAMIC_CONTENT_SCALING)

@end

#endif // PLATFORM(COCOA)
