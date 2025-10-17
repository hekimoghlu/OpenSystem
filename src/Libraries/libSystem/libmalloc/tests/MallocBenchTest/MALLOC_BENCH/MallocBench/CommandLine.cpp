/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#include "CommandLine.h"
#include <getopt.h>
#include <iostream>

struct option CommandLine::longOptions[] =
{
    {"benchmark", required_argument, 0, 'b'},
    {"detailed-report", no_argument, 0, 'd'},
    {"no-warmup", no_argument, 0, 'n' },
    {"parallel", no_argument, 0, 'p'},
    {"heap", required_argument, 0, 'h'},
    {"runs", required_argument, 0, 'r'},
    {"use-thread-id", no_argument, 0, 't'},
    {0, 0, 0, 0}
};

CommandLine::CommandLine(int argc, char** argv)
    : m_argc(argc)
    , m_argv(argv)
    , m_detailedReport(false)
    , m_isParallel(false)
    , m_useThreadID(false)
    , m_warmUp(true)
    , m_heapSize(0)
    , m_runs(8)
{
    int optionIndex = 0;
    int ch;
    while ((ch = getopt_long(argc, argv, "b:dnph:r:t", longOptions, &optionIndex)) != -1) {
        switch (ch)
        {
            case 'b':
                m_benchmarkName = optarg;
                break;

            case 'd':
                m_detailedReport = true;
                break;
                
            case 'n':
                m_warmUp = false;
                break;

            case 'p':
                m_isParallel = true;
                break;
                
            case 'h':
                m_heapSize = atoi(optarg) * 1024 * 1024;
                break;

            case 'r':
                m_runs = atoi(optarg);
                break;

            case 't':
                m_useThreadID = true;
                break;
                
            default:
                break;
        }
    }
}

void CommandLine::printUsage()
{
    std::string fullPath(m_argv[0]);
    size_t pos = fullPath.find_last_of("/") + 1;
    std::string program = fullPath.substr(pos);
    std::cout << "Usage: " << program << " --benchmark benchmark_name [--parallel ] [--use-thread-id ] [--detailed-report] [--no-warmup] [--runs count] [--heap MB ]" << std::endl;
}
