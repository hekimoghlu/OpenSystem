/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#include <IOKit/IOKitLib.h>
#include <ApplicationServices/ApplicationServices.h>
#include <IOKit/i2c/IOI2CInterface.h>

#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#define MAX_DISPLAYS    16

int main( int argc, char * argv[] )
{
    CGDirectDisplayID dispids[MAX_DISPLAYS];
    CGDisplayCount    ndid, idx;
    CGError err;
    char c;

    CGGetActiveDisplayList(MAX_DISPLAYS, dispids, &ndid);

    for (idx = 0; idx < ndid; idx++)
    {
        err = CGDisplayCaptureWithOptions(dispids[idx], kCGCaptureNoFill);
        printf("CGDisplayCapture(%x) %d\n", dispids[idx], err);
        CGDisplayHideCursor(dispids[idx]);
    }

    c = getchar();

    for (idx = 0; idx < ndid; idx++)
    {
        err = CGDisplayRelease(dispids[idx]);
        printf("CGDisplayRelease(%x) %d\n", dispids[idx], err);
        CGDisplayShowCursor(dispids[idx]);
    }

    return (0);
}
