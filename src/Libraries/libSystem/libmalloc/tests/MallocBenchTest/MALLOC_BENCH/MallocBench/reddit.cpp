/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 25, 2025.
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
#include "reddit.h"
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

void benchmark_reddit(CommandLine& commandLine)
{
    size_t times = 6;

    Interpreter interpreter("reddit.ops");
    for (size_t i = 0; i < times; ++i)
        interpreter.run();

    if (commandLine.detailedReport())
        interpreter.detailedReport();
}

void benchmark_reddit_memory_warning(CommandLine& commandLine)
{
    size_t times = 1;

    bool shouldFreeAllObjects = false;
    Interpreter interpreter("reddit_memory_warning.ops", shouldFreeAllObjects);
    for (size_t i = 0; i < times; ++i)
        interpreter.run();

    if (commandLine.detailedReport())
        interpreter.detailedReport();
}
