/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
#include "WebXRPose.h"

#if ENABLE(WEBXR)

#include "WebXRRigidTransform.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(WebXRPose);

Ref<WebXRPose> WebXRPose::create(Ref<WebXRRigidTransform>&& transform, bool emulatedPosition)
{
    return adoptRef(*new WebXRPose(WTFMove(transform), emulatedPosition));
}

WebXRPose::WebXRPose(Ref<WebXRRigidTransform>&& transform, bool emulatedPosition)
    : m_transform(WTFMove(transform)), m_emulatedPosition(emulatedPosition)
{
}

WebXRPose::~WebXRPose() = default;

const WebXRRigidTransform& WebXRPose::transform() const
{
    return m_transform;
}

bool WebXRPose::emulatedPosition() const
{
    return m_emulatedPosition;
}

} // namespace WebCore

#endif // ENABLE(WEBXR)
