/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 10, 2022.
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

#if USE(LIBWEBRTC)

#include "VideoDecoder.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class LibWebRTCVPXInternalVideoDecoder;

class LibWebRTCVPXVideoDecoder : public VideoDecoder {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCVPXVideoDecoder);
public:
    enum class Type {
        VP8,
        VP9,
        VP9_P2,
#if ENABLE(AV1)
        AV1
#endif
    };
    static void create(Type, const Config&, CreateCallback&&, OutputCallback&&);

    ~LibWebRTCVPXVideoDecoder();

private:
    LibWebRTCVPXVideoDecoder(Type, const Config&, OutputCallback&&);

    Ref<DecodePromise> decode(EncodedFrame&&) final;
    Ref<GenericPromise> flush() final;
    void reset() final;
    void close() final;

    Ref<LibWebRTCVPXInternalVideoDecoder> m_internalDecoder;
};

}

#endif // USE(LIBWEBRTC)
