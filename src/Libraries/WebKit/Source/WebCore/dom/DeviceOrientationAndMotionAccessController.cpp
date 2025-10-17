/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 13, 2022.
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
#include "DeviceOrientationAndMotionAccessController.h"

#if ENABLE(DEVICE_ORIENTATION)

#include "Chrome.h"
#include "ChromeClient.h"
#include "Document.h"
#include "DocumentLoader.h"
#include "FrameLoader.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "Page.h"
#include "UserGestureIndicator.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DeviceOrientationAndMotionAccessController);

DeviceOrientationAndMotionAccessController::DeviceOrientationAndMotionAccessController(Document& topDocument)
    : m_topDocument(topDocument)
{
}

DeviceOrientationOrMotionPermissionState DeviceOrientationAndMotionAccessController::accessState(const Document& document) const
{
    auto iterator = m_accessStatePerOrigin.find(document.securityOrigin().data());
    if (iterator != m_accessStatePerOrigin.end())
        return iterator->value;

    // Check per-site setting.
    Ref topDocument = m_topDocument.get();
    if (&document == topDocument.ptr() || document.protectedSecurityOrigin()->isSameOriginAs(topDocument->protectedSecurityOrigin())) {
        RefPtr frame = topDocument->frame();
        if (RefPtr documentLoader = frame ? frame->loader().documentLoader() : nullptr)
            return documentLoader->deviceOrientationAndMotionAccessState();
    }

    return DeviceOrientationOrMotionPermissionState::Prompt;
}

void DeviceOrientationAndMotionAccessController::shouldAllowAccess(const Document& document, Function<void(DeviceOrientationOrMotionPermissionState)>&& callback)
{
    RefPtr page = document.page();
    RefPtr frame = document.frame();
    if (!page || !frame)
        return callback(DeviceOrientationOrMotionPermissionState::Denied);

    auto accessState = this->accessState(document);
    if (accessState != DeviceOrientationOrMotionPermissionState::Prompt)
        return callback(accessState);

    bool mayPrompt = UserGestureIndicator::processingUserGesture(&document);
    page->chrome().client().shouldAllowDeviceOrientationAndMotionAccess(document.protectedFrame().releaseNonNull(), mayPrompt, [this, weakThis = WeakPtr { *this }, securityOrigin = Ref { document.securityOrigin() }, callback = WTFMove(callback)](DeviceOrientationOrMotionPermissionState permissionState) mutable {
        if (!weakThis)
            return;

        m_accessStatePerOrigin.set(securityOrigin->data(), permissionState);
        callback(permissionState);
        if (!weakThis)
            return;

        if (permissionState != DeviceOrientationOrMotionPermissionState::Granted)
            return;

        for (RefPtr<Frame> frame = m_topDocument->frame(); frame && frame->window(); frame = frame->tree().traverseNext()) {
            RefPtr localFrame = dynamicDowncast<LocalFrame>(frame);
            if (!localFrame)
                continue;
            RefPtr window = localFrame->window();
            window->startListeningForDeviceOrientationIfNecessary();
            window->startListeningForDeviceMotionIfNecessary();
        }
    });
}

} // namespace WebCore

#endif // ENABLE(DEVICE_ORIENTATION)
