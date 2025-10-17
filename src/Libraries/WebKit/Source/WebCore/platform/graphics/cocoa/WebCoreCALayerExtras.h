/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 4, 2021.
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
#import <QuartzCore/QuartzCore.h>

#ifdef __cplusplus
#import "FloatPoint.h"
#import "ScrollingNodeID.h"
#import <wtf/Vector.h>
#endif

@interface CALayer (WebCoreCALayerExtras)

+ (CALayer *)_web_renderLayerWithContextID:(uint32_t)contextID shouldPreserveFlip:(BOOL)preservesFlip;

- (void)web_disableAllActions;
- (void)_web_setLayerBoundsOrigin:(CGPoint)origin;
- (void)_web_setLayerTopLeftPosition:(CGPoint)position;
- (BOOL)_web_maskContainsPoint:(CGPoint)point;
- (BOOL)_web_maskMayIntersectRect:(CGRect)rect;
- (void)_web_clearContents;

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
- (void)_web_clearDynamicContentScalingDisplayListIfNeeded;
#endif

@end

#ifdef __cplusplus

namespace WebCore {

using LayerAndPoint = std::pair<RetainPtr<CALayer>, FloatPoint>;
WEBCORE_EXPORT void collectDescendantLayersAtPoint(Vector<LayerAndPoint, 16>& layersAtPoint, CALayer *parent, CGPoint, const std::function<bool(CALayer *, CGPoint localPoint)>& pointInLayerFunction = { });

WEBCORE_EXPORT Vector<LayerAndPoint, 16> layersAtPointToCheckForScrolling(std::function<bool(CALayer*, CGPoint)> layerEventRegionContainsPoint, std::function<std::optional<ScrollingNodeID>(CALayer*)> scrollingNodeIDForLayer, CALayer*, const FloatPoint&, bool& hasAnyNonInteractiveScrollingLayers);

} // namespace WebCore

#endif // __cplusplus
