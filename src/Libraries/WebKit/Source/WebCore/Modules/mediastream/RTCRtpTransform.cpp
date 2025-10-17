/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#include "RTCRtpTransform.h"

#if ENABLE(WEB_RTC)

#include "RTCRtpReceiver.h"
#include "RTCRtpSender.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RTCRtpTransform);

std::unique_ptr<RTCRtpTransform> RTCRtpTransform::from(std::optional<Internal>&& internal)
{
    if (!internal)
        return nullptr;
    return makeUnique<RTCRtpTransform>(WTFMove(*internal));
}

RTCRtpTransform::RTCRtpTransform(Internal&& transform)
    : m_transform(WTFMove(transform))
{
}

RTCRtpTransform::~RTCRtpTransform()
{
    clearBackend();
}

bool RTCRtpTransform::isAttached() const
{
    return WTF::switchOn(m_transform, [&](const RefPtr<RTCRtpSFrameTransform>& sframeTransform) {
        return sframeTransform->isAttached();
    }, [&](const RefPtr<RTCRtpScriptTransform>& scriptTransform) {
        return scriptTransform->isAttached();
    });
}

void RTCRtpTransform::attachToReceiver(RTCRtpReceiver& receiver, RTCRtpTransform* previousTransform)
{
    ASSERT(!isAttached());

    if (previousTransform)
        m_backend = previousTransform->takeBackend();
    else if (auto* backend = receiver.backend())
        m_backend = backend->rtcRtpTransformBackend();

    if (!m_backend)
        return;

    switchOn(m_transform, [&](RefPtr<RTCRtpSFrameTransform>& sframeTransform) {
        sframeTransform->initializeBackendForReceiver(*m_backend);
    }, [&](RefPtr<RTCRtpScriptTransform>& scriptTransform) {
        scriptTransform->initializeBackendForReceiver(*m_backend);
    });
}

void RTCRtpTransform::attachToSender(RTCRtpSender& sender, RTCRtpTransform* previousTransform)
{
    ASSERT(!isAttached());

    if (previousTransform)
        m_backend = previousTransform->takeBackend();
    else if (auto* backend = sender.backend())
        m_backend = backend->rtcRtpTransformBackend();

    if (!m_backend)
        return;

    switchOn(m_transform, [&](RefPtr<RTCRtpSFrameTransform>& sframeTransform) {
        sframeTransform->initializeBackendForSender(*m_backend);
    }, [&](RefPtr<RTCRtpScriptTransform>& scriptTransform) {
        scriptTransform->initializeBackendForSender(*m_backend);
        if (previousTransform)
            previousTransform->backendTransferedToNewTransform();
    });
}

void RTCRtpTransform::backendTransferedToNewTransform()
{
    switchOn(m_transform, [&](RefPtr<RTCRtpSFrameTransform>&) {
    }, [&](RefPtr<RTCRtpScriptTransform>& scriptTransform) {
        scriptTransform->backendTransferedToNewTransform();
    });
}

void RTCRtpTransform::clearBackend()
{
    if (!m_backend)
        return;

    switchOn(m_transform, [&](RefPtr<RTCRtpSFrameTransform>& sframeTransform) {
        sframeTransform->willClearBackend(*m_backend);
    }, [&](RefPtr<RTCRtpScriptTransform>& scriptTransform) {
        scriptTransform->willClearBackend(*m_backend);
    });

    m_backend = nullptr;
}

void RTCRtpTransform::detachFromReceiver(RTCRtpReceiver&)
{
    clearBackend();
}

void RTCRtpTransform::detachFromSender(RTCRtpSender&)
{
    clearBackend();
}

bool operator==(const RTCRtpTransform& a, const RTCRtpTransform& b)
{
    return WTF::switchOn(a.m_transform, [&](const RefPtr<RTCRtpSFrameTransform>& sframeTransformA) {
        return WTF::switchOn(b.m_transform, [&](const RefPtr<RTCRtpSFrameTransform>& sframeTransformB) {
            return sframeTransformA.get() == sframeTransformB.get();
        }, [&](const RefPtr<RTCRtpScriptTransform>&) {
            return false;
        });
    }, [&](const RefPtr<RTCRtpScriptTransform>& scriptTransformA) {
        return WTF::switchOn(b.m_transform, [&](const RefPtr<RTCRtpSFrameTransform>&) {
            return false;
        }, [&](const RefPtr<RTCRtpScriptTransform>& scriptTransformB) {
            return scriptTransformA.get() == scriptTransformB.get();
        });
    });
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
