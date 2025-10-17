/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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

#if PLATFORM(MAC)

namespace WebCore {

class FloatRect;
class FloatSize;
class GraphicsContext;
class ImageBuffer;
class IntSize;

struct ControlStyle;

template<typename> class RectEdges;
using IntOutsets = RectEdges<int>;

namespace SwitchMacUtilities {

IntSize cellSize(NSControlSize);
FloatSize visualCellSize(IntSize, const ControlStyle&);
IntOutsets cellOutsets(NSControlSize);
IntOutsets visualCellOutsets(NSControlSize, bool);
FloatRect rectForBounds(const FloatRect&);
NSString *coreUISizeForControlSize(const NSControlSize);
float easeInOut(float);
FloatRect rectWithTransposedSize(const FloatRect&, bool);
FloatRect trackRectForBounds(const FloatRect&, const FloatSize&);
void rotateContextForVerticalWritingMode(GraphicsContext&, const FloatRect&);
RefPtr<ImageBuffer> trackMaskImage(GraphicsContext&, FloatSize, float, bool, NSString *);

} // namespace SwitchMacUtilities

} // namespace WebCore

#endif // PLATFORM(MAC)
