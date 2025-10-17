/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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

#if ENABLE(WEBXR)

#include "ExceptionOr.h"
#include "JSValueInWrappedObject.h"
#include "TransformationMatrix.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

struct DOMPointInit;
class DOMPointReadOnly;

class WebXRRigidTransform : public RefCountedAndCanMakeWeakPtr<WebXRRigidTransform> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(WebXRRigidTransform, WEBCORE_EXPORT);
public:
    static Ref<WebXRRigidTransform> create();
    static Ref<WebXRRigidTransform> create(const TransformationMatrix&);
    WEBCORE_EXPORT static ExceptionOr<Ref<WebXRRigidTransform>> create(const DOMPointInit&, const DOMPointInit&);
    WEBCORE_EXPORT ~WebXRRigidTransform();

    const DOMPointReadOnly& position() const;
    const DOMPointReadOnly& orientation() const;
    const Float32Array& matrix();
    const WebXRRigidTransform& inverse();
    const TransformationMatrix& rawTransform() const;

    JSValueInWrappedObject& cachedMatrix() { return m_cachedMatrix; }

private:
    WebXRRigidTransform(const DOMPointInit&, const DOMPointInit&);
    WebXRRigidTransform(const TransformationMatrix&);

    Ref<DOMPointReadOnly> m_position;
    Ref<DOMPointReadOnly> m_orientation;
    TransformationMatrix m_rawTransform;
    RefPtr<Float32Array> m_matrix;
    RefPtr<WebXRRigidTransform> m_inverse;
    WeakPtr<WebXRRigidTransform> m_parentInverse;
    JSValueInWrappedObject m_cachedMatrix;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
