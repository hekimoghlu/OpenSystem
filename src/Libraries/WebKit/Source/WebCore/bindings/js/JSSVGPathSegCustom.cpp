/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#include "JSSVGPathSeg.h"

#include "JSDOMBinding.h"
#include "JSSVGPathSegArcAbs.h"
#include "JSSVGPathSegArcRel.h"
#include "JSSVGPathSegClosePath.h"
#include "JSSVGPathSegCurvetoCubicAbs.h"
#include "JSSVGPathSegCurvetoCubicRel.h"
#include "JSSVGPathSegCurvetoCubicSmoothAbs.h"
#include "JSSVGPathSegCurvetoCubicSmoothRel.h"
#include "JSSVGPathSegCurvetoQuadraticAbs.h"
#include "JSSVGPathSegCurvetoQuadraticRel.h"
#include "JSSVGPathSegCurvetoQuadraticSmoothAbs.h"
#include "JSSVGPathSegCurvetoQuadraticSmoothRel.h"
#include "JSSVGPathSegLinetoAbs.h"
#include "JSSVGPathSegLinetoRel.h"
#include "JSSVGPathSegLinetoHorizontalAbs.h"
#include "JSSVGPathSegLinetoHorizontalRel.h"
#include "JSSVGPathSegLinetoVerticalAbs.h"
#include "JSSVGPathSegLinetoVerticalRel.h"
#include "JSSVGPathSegMovetoAbs.h"
#include "JSSVGPathSegMovetoRel.h"
#include "SVGPathSeg.h"

namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<SVGPathSeg>&& object)
{
    switch (object->pathSegType()) {
    case SVGPathSegType::ClosePath:
        return createWrapper<SVGPathSegClosePath>(globalObject, WTFMove(object));
    case SVGPathSegType::MoveToAbs:
        return createWrapper<SVGPathSegMovetoAbs>(globalObject, WTFMove(object));
    case SVGPathSegType::MoveToRel:
        return createWrapper<SVGPathSegMovetoRel>(globalObject, WTFMove(object));
    case SVGPathSegType::LineToAbs:
        return createWrapper<SVGPathSegLinetoAbs>(globalObject, WTFMove(object));
    case SVGPathSegType::LineToRel:
        return createWrapper<SVGPathSegLinetoRel>(globalObject, WTFMove(object));
    case SVGPathSegType::CurveToCubicAbs:
        return createWrapper<SVGPathSegCurvetoCubicAbs>(globalObject, WTFMove(object));
    case SVGPathSegType::CurveToCubicRel:
        return createWrapper<SVGPathSegCurvetoCubicRel>(globalObject, WTFMove(object));
    case SVGPathSegType::CurveToQuadraticAbs:
        return createWrapper<SVGPathSegCurvetoQuadraticAbs>(globalObject, WTFMove(object));
    case SVGPathSegType::CurveToQuadraticRel:
        return createWrapper<SVGPathSegCurvetoQuadraticRel>(globalObject, WTFMove(object));
    case SVGPathSegType::ArcAbs:
        return createWrapper<SVGPathSegArcAbs>(globalObject, WTFMove(object));
    case SVGPathSegType::ArcRel:
        return createWrapper<SVGPathSegArcRel>(globalObject, WTFMove(object));
    case SVGPathSegType::LineToHorizontalAbs:
        return createWrapper<SVGPathSegLinetoHorizontalAbs>(globalObject, WTFMove(object));
    case SVGPathSegType::LineToHorizontalRel:
        return createWrapper<SVGPathSegLinetoHorizontalRel>(globalObject, WTFMove(object));
    case SVGPathSegType::LineToVerticalAbs:
        return createWrapper<SVGPathSegLinetoVerticalAbs>(globalObject, WTFMove(object));
    case SVGPathSegType::LineToVerticalRel:
        return createWrapper<SVGPathSegLinetoVerticalRel>(globalObject, WTFMove(object));
    case SVGPathSegType::CurveToCubicSmoothAbs:
        return createWrapper<SVGPathSegCurvetoCubicSmoothAbs>(globalObject, WTFMove(object));
    case SVGPathSegType::CurveToCubicSmoothRel:
        return createWrapper<SVGPathSegCurvetoCubicSmoothRel>(globalObject, WTFMove(object));
    case SVGPathSegType::CurveToQuadraticSmoothAbs:
        return createWrapper<SVGPathSegCurvetoQuadraticSmoothAbs>(globalObject, WTFMove(object));
    case SVGPathSegType::CurveToQuadraticSmoothRel:
        return createWrapper<SVGPathSegCurvetoQuadraticSmoothRel>(globalObject, WTFMove(object));
    case SVGPathSegType::Unknown:
    default:
        return createWrapper<SVGPathSeg>(globalObject, WTFMove(object));
    }
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, SVGPathSeg& object)
{
    return wrap(lexicalGlobalObject, globalObject, object);
}

}
