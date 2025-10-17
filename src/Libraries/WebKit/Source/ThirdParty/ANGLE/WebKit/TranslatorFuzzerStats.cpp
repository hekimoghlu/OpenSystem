/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
// The fuzzer RSS grows. This tries to investigate the corpus.
// ~/Build/Release/ANGLETranslatorFuzzerStats CorpusNew3/* |grep -v "diff: 0" > rss-increases.txt
// sort rss-increases.txt |less
// ~/Build/Release/ANGLETranslatorFuzzerStats CorpusNew3/0d994bf8eb288db4fb26be209a3ae3e048f989e

#include <iostream>
#include <fstream>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size);

// In order to link with the fuzzer that uses LLVMFuzzerMutate, provide a dummy symbol for the function.
extern "C" size_t LLVMFuzzerMutate(uint8_t*, size_t, size_t)
{
    exit(1);
    return 0;
}

static size_t getRSSKB()
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage))
        return 0;
    return static_cast<size_t>(usage.ru_maxrss >> 10L);
}

static const int iterations = 100;

int main(int argc, const char * argv[])
{
    std::vector<uint8_t> fileData;
    for (int j = 0; j < iterations; ++j) {
        for (int i = 1; i < argc; ++i) {
            std::streampos fileSize;
            {
                std::ifstream file { argv[i], std::ios::binary };
                file.seekg(0, std::ios::end);
                fileSize = file.tellg();
                file.seekg(0, std::ios::beg);
                if (fileData.size() < static_cast<size_t>(fileSize))
                    fileData.resize(static_cast<size_t>(fileSize));

                file.read(reinterpret_cast<char*>(&fileData[0]), fileSize);
            }
            size_t rss = getRSSKB();
            LLVMFuzzerTestOneInput(&fileData[0], fileSize);
            size_t newRSS = getRSSKB();
            if (iterations == 1 || i != 0)
                std::cout << argv[i] << " rss: " << rss << "kb rss diff: " << newRSS - rss << "kb" << std::endl;
        }
    }
    return 0;
}
