/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#include <WebCore/GtkVersioning.h>
#include <libintl.h>

#if USE(GSTREAMER)
#include <WebCore/GStreamerCommon.h>
#endif

#if USE(GCRYPT)
#include <pal/crypto/gcrypt/Initialization.h>
#endif

#if USE(SKIA)
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <skia/core/SkGraphics.h>
#if USE(SKIA_OPENTYPE_SVG)
#include <skia/modules/svg/SkSVGOpenTypeSVGDecoder.h>
#endif
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END
#endif

#if USE(SYSPROF_CAPTURE)
#include <wtf/SystemTracing.h>
#endif

namespace WebKit {
using namespace WebCore;

class WebProcessMainGtk final: public AuxiliaryProcessMainBase<WebProcess> {
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
            g_usleep(30 * G_USEC_PER_SEC);
#endif

        gtk_init(nullptr, nullptr);

        bindtextdomain(GETTEXT_PACKAGE, LOCALEDIR);
        bind_textdomain_codeset(GETTEXT_PACKAGE, "UTF-8");

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
#if USE(ATSPI)
    // Disable ATK/GTK accessibility support in the WebProcess.
#if USE(GTK4)
    g_setenv("GTK_A11Y", "none", TRUE);
#else
    g_setenv("NO_AT_BRIDGE", "1", TRUE);
#endif
#endif

    // Ignore the GTK_THEME environment variable, the theme is always set by the UI process now.
    // This call needs to happen before any threads begin execution
    unsetenv("GTK_THEME");

    return AuxiliaryProcessMain<WebProcessMainGtk>(argc, argv);
}

} // namespace WebKit
