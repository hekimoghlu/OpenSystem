/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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
#pragma once

#include <WebKit/WKBase.h>
#include <WebKit/WKGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

WK_EXPORT void WKRenderLayerGetTypeID(void);

WK_EXPORT void WKRenderLayerGetRenderer(void);

WK_EXPORT void WKRenderLayerCopyRendererName(void);

WK_EXPORT void WKRenderLayerCopyElementTagName(void);
WK_EXPORT void WKRenderLayerCopyElementID(void);
WK_EXPORT void WKRenderLayerGetElementClassNames(void);

WK_EXPORT void WKRenderLayerGetAbsoluteBounds(void);

WK_EXPORT void WKRenderLayerIsClipping(void);
WK_EXPORT void WKRenderLayerIsClipped(void);
WK_EXPORT void WKRenderLayerIsReflection(void);

WK_EXPORT void WKRenderLayerGetCompositingLayerType(void);
WK_EXPORT void WKRenderLayerGetBackingStoreMemoryEstimate(void);

WK_EXPORT void WKRenderLayerGetNegativeZOrderList(void);
WK_EXPORT void WKRenderLayerGetNormalFlowList(void);
WK_EXPORT void WKRenderLayerGetPositiveZOrderList(void);

WK_EXPORT void WKRenderLayerGetFrameContentsLayer(void);

#ifdef __cplusplus
}
#endif
