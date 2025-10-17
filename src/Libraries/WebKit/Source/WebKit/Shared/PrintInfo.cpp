/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#include "PrintInfo.h"

namespace WebKit {

PrintInfo::PrintInfo(float pageSetupScaleFactor, float availablePaperWidth, float availablePaperHeight, WebCore::FloatBoxExtent margin
#if PLATFORM(IOS_FAMILY)
    , bool snapshotFirstPage
#endif
#if PLATFORM(GTK)
    , GRefPtr<GtkPrintSettings>&& printSettings, GRefPtr<GtkPageSetup>&& pageSetup, PrintMode printMode
#endif
    )
    : pageSetupScaleFactor(pageSetupScaleFactor)
    , availablePaperWidth(availablePaperWidth)
    , availablePaperHeight(availablePaperHeight)
    , margin(margin)
#if PLATFORM(IOS_FAMILY)
    , snapshotFirstPage(snapshotFirstPage)
#endif
#if PLATFORM(GTK)
    , printSettings(WTFMove(printSettings))
    , pageSetup(WTFMove(pageSetup))
    , printMode(printMode)
#endif
{
}

}
