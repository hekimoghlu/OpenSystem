/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#import <XCTest/XCTest.h>
#import <string>
#import <fstream>
#import <filesystem>
#import <vector>

extern "C" int LLVMFuzzerTestOneInput(const unsigned char *data, size_t size);

// In order to link with the fuzzer that uses LLVMFuzzerMutate, provide a dummy symbol for the function.
extern "C" size_t LLVMFuzzerMutate(uint8_t*, size_t, size_t)
{
    exit(1);
    return 0;
}


@interface TranslatorFuzzerCoverage : XCTestCase

@end

@implementation TranslatorFuzzerCoverage

- (void)setUp {

}

- (void)tearDown {

}

- (void)testFuzzerCorpusCoverage {
    const char* corpusPathEnv = getenv("ANGLE_TRANSLATOR_FUZZER_CORPUS_PATH");
    std::string corpusPath = corpusPathEnv ? corpusPathEnv : "corpus";
    for (auto& fileEntry : std::filesystem::directory_iterator { corpusPath })
    {
        std::ifstream file { fileEntry.path(), std::ios::binary };
        file.seekg(0, std::ios::end);
        std::streampos fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> fileData(fileSize);
        file.read(reinterpret_cast<char*>(&fileData[0]), fileSize);
        LLVMFuzzerTestOneInput(&fileData[0], fileData.size());
        XCTAssertTrue(true, @"Case done: %s", fileEntry.path().c_str());
    }
}

@end
