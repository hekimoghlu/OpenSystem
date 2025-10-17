/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 7, 2022.
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
#include "Benchmark.h"
#include "CommandLine.h"
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using namespace std;

int main(int argc, char** argv)
{
    CommandLine commandLine(argc, argv);
    if (!commandLine.isValid()) {
        commandLine.printUsage();
        exit(1);
    }

    Benchmark benchmark(commandLine);
    if (!benchmark.isValid()) {
        cout << "Invalid benchmark: " << commandLine.benchmarkName() << endl << endl;
        benchmark.printBenchmarks();
        exit(1);
    }

    string parallel = commandLine.isParallel() ? string(" [ parallel ]") : string(" [ not parallel ]");
    string threaded = commandLine.useThreadID() ? string(" [ use-thread-id ]") : string(" [ don't use-thread-id ]");

    stringstream runs;
    runs << " [ runs: " << commandLine.runs() << " ]";

    stringstream heapSize;
    if (commandLine.heapSize())
        heapSize << " [ heap: " << commandLine.heapSize() / 1024 / 1024 << "MB ]";
    else
        heapSize << " [ heap: 0MB ]";

    cout << "Running " << commandLine.benchmarkName() << parallel << threaded << heapSize.str() << runs.str() << "..." << endl;
    benchmark.run();
    benchmark.printReport();
        
    return 0;
}
