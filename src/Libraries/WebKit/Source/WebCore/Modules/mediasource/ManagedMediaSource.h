/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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

#if ENABLE(MEDIA_SOURCE)

#include "MediaSource.h"
#include "Timer.h"
#include <optional>

namespace WebCore {

class ManagedMediaSource final : public MediaSource {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ManagedMediaSource);
public:
    static Ref<ManagedMediaSource> create(ScriptExecutionContext&, MediaSourceInit&&);
    ~ManagedMediaSource();

    enum class PreferredQuality { Low, Medium, High };
    ExceptionOr<PreferredQuality> quality() const;

    static bool isTypeSupported(ScriptExecutionContext&, const String& type);

    bool streaming() const override { return m_streaming; }
    bool streamingAllowed() const { return m_streamingAllowed; }

    bool isManaged() const final { return true; }

private:
    ManagedMediaSource(ScriptExecutionContext&, MediaSourceInit&&);
    void monitorSourceBuffers() final;
    void elementDetached() final;
    void setStreaming(bool);
    void streamingTimerFired();
    void ensurePrefsRead();

    bool m_streaming { false };
    std::optional<double> m_lowThreshold;
    std::optional<double> m_highThreshold;
    Timer m_streamingTimer;
    bool m_streamingAllowed { true };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ManagedMediaSource)
    static bool isType(const WebCore::MediaSource& mediaSource) { return mediaSource.isManaged(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(MEDIA_SOURCE)
