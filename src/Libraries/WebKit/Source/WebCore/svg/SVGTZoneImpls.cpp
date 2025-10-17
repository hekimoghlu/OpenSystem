/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#include "config.h"

#include "ElementInlines.h"
#include "SVGAnimatedPropertyAnimatorImpl.h"
#include "SVGAnimatedPropertyPairAnimatorImpl.h"
#include "SVGDecoratedProperty.h"
#include "SVGElementRareData.h"
#include "SVGImageClients.h"
#include "SVGPathByteStream.h"
#include "SVGPathConsumer.h"
#include "SVGPathSource.h"
#include "SVGPropertyAnimatorFactory.h"
#include "SVGValuePropertyAnimatorImpl.h"

#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedAngleAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedAngleOrientAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedBooleanAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedIntegerAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedIntegerPairAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedLengthAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedLengthListAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedNumberAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedNumberListAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedNumberPairAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedOrientTypeAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedPathSegListAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedPointListAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedPreserveAspectRatioAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedRectAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedStringAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAnimatedTransformListAnimator);

WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGElementRareData);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGImageChromeClient);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGLengthAnimator);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGPathByteStream);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGPathConsumer);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGPathSource);
WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGPropertyAnimatorFactory);

} // namespace WebCore
