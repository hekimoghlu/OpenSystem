/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
#include "WebProcessMain.h"

#include "AuxiliaryProcessMain.h"
#include "WebProcess.h"
#include <glib.h>

#if USE(GCRYPT)
#include <pal/crypto/gcrypt/Initialization.h>
#endif

#if USE(GSTREAMER)
#include <WebCore/GStreamerCommon.h>
#endif

#if USE(SKIA)
#include <skia/core/SkGraphics.h>
#endif

#if USE(SYSPROF_CAPTURE)
#include <wtf/SystemTracing.h>
#if USE(SKIA_OPENTYPE_SVG)
#include <skia/modules/svg/SkSVGOpenTypeSVGDecoder.h>
#endif
#endif

namespace WebKit {
using namespace WebCore;

class WebProcessMainWPE final : public AuxiliaryProcessMainBase<WebProcess> {
public:
    bool platformInitialize() override
    {
#if USE(SYSPROF_CAPTURE)
        SysprofAnnotator::createIfNeeded("WebKit (Web)"_s);
#endif

#if USE(GCRYPT)
        PAL::GCrypt::initialize();
#endif

#if USE(SKIA)
        SkGraphics::Init();
#if USE(SKIA_OPENTYPE_SVG)
        SkGraphics::SetOpenTypeSVGDecoderFactory(SkSVGOpenTypeSVGDecoder::Make);
#endif
#endif

#if ENABLE(DEVELOPER_MODE)
        if (g_getenv("WEBKIT2_PAUSE_WEB_PROCESS_ON_LAUNCH"))
            WTF::sleep(30_s);
#endif

        // Required for GStreamer initialization.
        // FIXME: This should be probably called in other processes as well.
        g_set_prgname("WPEWebProcess");

        return true;
    }

    void platformFinalize() override
    {
#if USE(GSTREAMER)
        deinitializeGStreamer();
#endif
    }
};

int WebProcessMain(int argc, char** argv)
{
    return AuxiliaryProcessMain<WebProcessMainWPE>(argc, argv);
}

} // namespace WebKit
