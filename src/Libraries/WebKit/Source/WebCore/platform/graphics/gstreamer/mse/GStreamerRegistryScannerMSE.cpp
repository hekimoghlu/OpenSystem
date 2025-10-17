/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#include "GStreamerRegistryScannerMSE.h"

#if USE(GSTREAMER) && ENABLE(MEDIA_SOURCE)
#include <wtf/NeverDestroyed.h>
#include <wtf/RuntimeApplicationChecks.h>

namespace WebCore {

static bool singletonInitialized = false;

GStreamerRegistryScannerMSE& GStreamerRegistryScannerMSE::singleton()
{
    static NeverDestroyed<GStreamerRegistryScannerMSE> sharedInstance;
    singletonInitialized = true;
    return sharedInstance;
}

void teardownGStreamerRegistryScannerMSE()
{
    if (!singletonInitialized)
        return;

    auto& scanner = GStreamerRegistryScannerMSE::singleton();
    scanner.teardown();
}

void GStreamerRegistryScannerMSE::getSupportedDecodingTypes(HashSet<String>& types)
{
    if (isInWebProcess())
        types = singleton().mimeTypeSet(GStreamerRegistryScanner::Configuration::Decoding);
    else
        types = GStreamerRegistryScannerMSE().mimeTypeSet(GStreamerRegistryScanner::Configuration::Decoding);
}


GStreamerRegistryScannerMSE::GStreamerRegistryScannerMSE()
    : GStreamerRegistryScanner::GStreamerRegistryScanner(true)
{
}

}
#endif
