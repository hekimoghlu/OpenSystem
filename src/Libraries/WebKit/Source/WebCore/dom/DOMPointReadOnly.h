/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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

#include "DOMPointInit.h"
#include "ExceptionOr.h"
#include "FloatPoint3D.h"
#include "ScriptWrappable.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct DOMMatrixInit;
class DOMPoint;
class WebCoreOpaqueRoot;

class DOMPointReadOnly : public ScriptWrappable, public RefCounted<DOMPointReadOnly> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(DOMPointReadOnly, WEBCORE_EXPORT);
public:
    static Ref<DOMPointReadOnly> create(double x, double y, double z, double w) { return adoptRef(*new DOMPointReadOnly(x, y, z, w)); }
    static Ref<DOMPointReadOnly> create(const DOMPointInit& init) { return create(init.x, init.y, init.z, init.w); }
    static Ref<DOMPointReadOnly> fromPoint(const DOMPointInit& init) { return create(init.x, init.y, init.z, init.w); }
    static Ref<DOMPointReadOnly> fromFloatPoint(const FloatPoint3D& p) { return create(p.x(), p.y(), p.z(), 1); }

    double x() const { return m_x; }
    double y() const { return m_y; }
    double z() const { return m_z; }
    double w() const { return m_w; }

    ExceptionOr<Ref<DOMPoint>> matrixTransform(DOMMatrixInit&&) const;

protected:
    DOMPointReadOnly(double x, double y, double z, double w)
        : m_x(x)
        , m_y(y)
        , m_z(z)
        , m_w(w)
    {
    }

    // Any of these can be NaN or Inf.
    double m_x;
    double m_y;
    double m_z;
    double m_w;
};

WebCoreOpaqueRoot root(DOMPointReadOnly*);

} // namespace WebCore

