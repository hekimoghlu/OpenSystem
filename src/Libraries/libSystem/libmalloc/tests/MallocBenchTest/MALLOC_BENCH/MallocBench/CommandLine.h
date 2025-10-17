/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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
#ifndef CommandLine_h
#define CommandLine_h

#include <string>

class CommandLine {
public:
    CommandLine(int argc, char** argv);

    bool isValid() { return m_benchmarkName.size(); }
    const std::string& benchmarkName() { return m_benchmarkName; }
    bool isParallel() { return m_isParallel; }
    bool useThreadID() { return m_useThreadID; }
    bool detailedReport() { return m_detailedReport; }
    bool warmUp() { return m_warmUp; }
    size_t heapSize() { return m_heapSize; }
    size_t runs() { return m_runs; }

    void printUsage();

private:
    static struct option longOptions[];

    int m_argc;
    char** m_argv;
    std::string m_benchmarkName;
    bool m_detailedReport;
    bool m_isParallel;
    bool m_useThreadID;
    bool m_warmUp;
    size_t m_heapSize;
    size_t m_runs;
};

#endif // CommandLine_h