/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 22, 2021.
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

#include "ExceptionOr.h"
#include "FloatPoint.h"

namespace WebCore {

class Gradient;
class ScriptExecutionContext;

class CanvasGradient : public RefCounted<CanvasGradient> {
public:
    static Ref<CanvasGradient> create(const FloatPoint& p0, const FloatPoint& p1);
    static Ref<CanvasGradient> create(const FloatPoint& p0, float r0, const FloatPoint& p1, float r1);
    static Ref<CanvasGradient> create(const FloatPoint& centerPoint, float angleInRadians);
    ~CanvasGradient();

    Gradient& gradient() { return m_gradient; }
    const Gradient& gradient() const { return m_gradient; }

    ExceptionOr<void> addColorStop(ScriptExecutionContext&, double value, const String& color);

private:
    CanvasGradient(const FloatPoint& p0, const FloatPoint& p1);
    CanvasGradient(const FloatPoint& p0, float r0, const FloatPoint& p1, float r1);
    CanvasGradient(const FloatPoint& centerPoint, float angleInRadians);

    Ref<Gradient> m_gradient;
};

} // namespace WebCore
