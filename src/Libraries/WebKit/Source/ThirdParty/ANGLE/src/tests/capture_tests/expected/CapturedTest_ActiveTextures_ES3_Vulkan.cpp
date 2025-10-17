/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 4, 2023.
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

#include "CapturedTest_ActiveTextures_ES3_Vulkan.h"
#include "trace_fixture.h"
#include "angle_trace_gl.h"

// Private Functions

void InitReplay(void)
{
    // binaryDataFileName = CapturedTest_ActiveTextures_ES3_Vulkan.angledata
    // maxClientArraySize = 72
    // maxClientArraySize = 72
    // readBufferSize = 276
    // resourceIDBufferSize = 0
    // contextID = 3
    // maxBuffer = 0
    // maxContext = 3
    // maxFenceNV = 0
    // maxFramebuffer = 2
    // maxImage = 0
    // maxMemoryObject = 0
    // maxProgramPipeline = 0
    // maxQuery = 0
    // maxRenderbuffer = 0
    // maxSampler = 0
    // maxSemaphore = 0
    // maxShaderProgram = 5
    // maxSurface = 1
    // maxSync = 0
    // maxTexture = 3
    // maxTransformFeedback = 0
    // maxVertexArray = 0
    // maxegl_Sync = 0
    InitializeReplay4("CapturedTest_ActiveTextures_ES3_Vulkan.angledata", 72, 276, 0, 3, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 5, 1, 0, 3, 0, 0, 0);
}

// Public Functions

void ReplayFrame(uint32_t frameIndex)
{
    switch (frameIndex)
    {
        case 1:
            ReplayFrame1();
            break;
        case 2:
            ReplayFrame2();
            break;
        case 3:
            ReplayFrame3();
            break;
        case 4:
            ReplayFrame4();
            break;
        default:
            break;
    }
}

