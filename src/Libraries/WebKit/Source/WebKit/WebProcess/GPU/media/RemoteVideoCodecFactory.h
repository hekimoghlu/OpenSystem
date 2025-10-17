/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 27, 2021.
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

#if USE(LIBWEBRTC) && PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(WEB_CODECS)

#include <WebCore/VideoDecoder.h>
#include <WebCore/VideoEncoder.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class WebProcess;

class RemoteVideoCodecFactory {
    WTF_MAKE_TZONE_ALLOCATED(RemoteVideoCodecFactory);
public:
    explicit RemoteVideoCodecFactory(WebProcess&);
    ~RemoteVideoCodecFactory();

private:
    static void createDecoder(const String&, const WebCore::VideoDecoder::Config&, WebCore::VideoDecoder::CreateCallback&&, WebCore::VideoDecoder::OutputCallback&&);
    static void createEncoder(const String&, const WebCore::VideoEncoder::Config&, WebCore::VideoEncoder::CreateCallback&&, WebCore::VideoEncoder::DescriptionCallback&&, WebCore::VideoEncoder::OutputCallback&&);
};

} // namespace WebKit

#endif // USE(LIBWEBRTC) && PLATFORM(COCOA) && ENABLE(GPU_PROCESS)
