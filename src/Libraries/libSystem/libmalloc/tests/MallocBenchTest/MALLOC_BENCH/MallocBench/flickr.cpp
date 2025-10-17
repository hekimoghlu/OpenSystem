/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#include "CPUCount.h"
#include "Interpreter.h"
#include "flickr.h"
#include <assert.h>
#include <cstddef>
#include <cstddef>
#include <cstdlib>
#include <errno.h>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

#include "mbmalloc.h"

void benchmark_flickr(CommandLine& commandLine)
{
    size_t times = 3;

    Interpreter interpreter("flickr.ops");
    for (size_t i = 0; i < times; ++i)
        interpreter.run();

    if (commandLine.detailedReport())
        interpreter.detailedReport();
}

void benchmark_flickr_memory_warning(CommandLine& commandLine)
{
    size_t times = 1;

    bool shouldFreeAllObjects = false;
    Interpreter interpreter("flickr_memory_warning.ops", shouldFreeAllObjects);
    for (size_t i = 0; i < times; ++i)
        interpreter.run();

    if (commandLine.detailedReport())
        interpreter.detailedReport();
}
