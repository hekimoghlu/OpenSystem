/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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

#include <WebCore/LengthBox.h>

#if USE(APPKIT)
OBJC_CLASS NSPrintInfo;
#elif PLATFORM(GTK)
#include <wtf/glib/GRefPtr.h>

typedef struct _GtkPageSetup GtkPageSetup;
typedef struct _GtkPrintJob GtkPrintJob;
typedef struct _GtkPrintSettings GtkPrintSettings;
#else
// FIXME: This should use the windows equivalent.
class NSPrintInfo;
#endif

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

struct PrintInfo {
    PrintInfo() = default;
#if PLATFORM(GTK)
    enum class PrintMode : uint8_t {
        Async,
        Sync
    };

#if HAVE(GTK_UNIX_PRINTING)
    explicit PrintInfo(GtkPrintJob*, PrintMode = PrintMode::Async);
#endif
#else
    explicit PrintInfo(NSPrintInfo *);
#endif
    PrintInfo(float pageSetupScaleFactor, float availablePaperWidth, float availablePaperHeight, WebCore::FloatBoxExtent margin
#if PLATFORM(IOS_FAMILY)
        , bool snapshotFirstPage
#endif
#if PLATFORM(GTK)
        , GRefPtr<GtkPrintSettings>&&, GRefPtr<GtkPageSetup>&&, PrintMode
#endif
        );


    // These values are in 'point' unit (and not CSS pixel).
    float pageSetupScaleFactor { 0 };
    float availablePaperWidth { 0 };
    float availablePaperHeight { 0 };
    WebCore::FloatBoxExtent margin;
#if PLATFORM(IOS_FAMILY)
    bool snapshotFirstPage { false };
#endif

#if PLATFORM(GTK)
    GRefPtr<GtkPrintSettings> printSettings;
    GRefPtr<GtkPageSetup> pageSetup;
    PrintMode printMode { PrintMode::Async };
#endif
};

} // namespace WebKit
