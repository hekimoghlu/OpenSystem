/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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
#include "GPUProcessMain.h"

#if ENABLE(GPU_PROCESS)

#include "AuxiliaryProcessMain.h"
#include "GPUProcess.h"
#include <Objbase.h>

namespace WebKit {

class GPUProcessMainWin final: public AuxiliaryProcessMainBase<GPUProcess> {
};

int GPUProcessMain(int argc, char** argv)
{
    // for DirectX
    HRESULT hr = ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    RELEASE_ASSERT(SUCCEEDED(hr));
    return AuxiliaryProcessMain<GPUProcessMainWin>(argc, argv);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
