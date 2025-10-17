/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 28, 2023.
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
#include "CairoPaintingEngine.h"

#if USE(CAIRO)
#include "CairoPaintingEngineBasic.h"
#include "CairoPaintingEngineThreaded.h"
#include <wtf/NumberOfCores.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {
namespace Cairo {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PaintingEngine);

std::unique_ptr<PaintingEngine> PaintingEngine::create()
{
#if PLATFORM(WPE) || USE(GTK4)
    unsigned numThreads = std::max(1, std::min(8, WTF::numberOfProcessorCores() / 2));
#else
    unsigned numThreads = 0;
#endif
    const char* numThreadsEnv = getenv("WEBKIT_CAIRO_PAINTING_THREADS");
    if (!numThreadsEnv)
        numThreadsEnv = getenv("WEBKIT_NICOSIA_PAINTING_THREADS");
    if (numThreadsEnv) {
        auto newValue = parseInteger<unsigned>(StringView::fromLatin1(numThreadsEnv));
        if (newValue && *newValue <= 8)
            numThreads = *newValue;
        else
            WTFLogAlways("The number of Cairo painting threads is not between 0 and 8. Using the default value %u\n", numThreads);
    }

    if (numThreads)
        return std::unique_ptr<PaintingEngine>(new PaintingEngineThreaded(numThreads));

    return std::unique_ptr<PaintingEngine>(new PaintingEngineBasic);
}

} // namespace Cairo
} // namespace WebCore

#endif // USE(CAIRO)
