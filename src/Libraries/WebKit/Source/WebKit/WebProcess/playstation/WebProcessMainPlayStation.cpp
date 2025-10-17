/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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

#if USE(SKIA) && !ENABLE(GPU_PROCESS)
#include <skia/core/SkGraphics.h>
#endif

namespace WebKit {

class WebProcessMainPlayStation final: public AuxiliaryProcessMainBase<WebProcess> {
public:
    bool platformInitialize() override
    {
#if USE(SKIA) && !ENABLE(GPU_PROCESS)
        SkGraphics::Init();
#endif
        return true;
    }
};

int WebProcessMain(int argc, char** argv)
{
    return AuxiliaryProcessMain<WebProcessMainPlayStation>(argc, argv);
}

} // namespace WebKit
