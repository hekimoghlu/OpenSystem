/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#ifndef _STRSRCHPERF_H
#define _STRSRCHPERF_H

#include "unicode/usearch.h"
#include "unicode/uperf.h"
#include <stdlib.h>
#include <stdio.h>

typedef void (*StrSrchFn)(UStringSearch* srch, const char16_t* src,int32_t srcLen, const char16_t* pttrn, int32_t pttrnLen, UErrorCode* status);

class StringSearchPerfFunction : public UPerfFunction {
private:
    StrSrchFn fn;
    const char16_t* src;
    int32_t srcLen;
    const char16_t* pttrn;
    int32_t pttrnLen;
    UStringSearch* srch;
    
public:
    void call(UErrorCode* status) override {
        (*fn)(srch, src, srcLen, pttrn, pttrnLen, status);
    }
    
    long getOperationsPerIteration() override {
        return static_cast<long>(srcLen);
    }
    
    StringSearchPerfFunction(StrSrchFn func, UStringSearch* search, const char16_t* source,int32_t sourceLen, const char16_t* pattern, int32_t patternLen) {
        fn = func;
        src = source;
        srcLen = sourceLen;
        pttrn = pattern;
        pttrnLen = patternLen;
        srch = search;
    }
};

class StringSearchPerformanceTest : public UPerfTest {
private:
    const char16_t* src;
    int32_t srcLen;
    char16_t* pttrn;
    int32_t pttrnLen;
    UStringSearch* srch;
    
public:
    StringSearchPerformanceTest(int32_t argc, const char *argv[], UErrorCode &status);
    ~StringSearchPerformanceTest();
    UPerfFunction* runIndexedTest(int32_t index, UBool exec, const char*& name, char* par = nullptr) override;
    UPerfFunction* Test_ICU_Forward_Search();
    UPerfFunction* Test_ICU_Backward_Search();
};


void ICUForwardSearch(UStringSearch *srch, const char16_t* source, int32_t sourceLen, const char16_t* pattern, int32_t patternLen, UErrorCode* status) {
    int32_t match;
    
    match = usearch_first(srch, status);
    while (match != USEARCH_DONE) {
        match = usearch_next(srch, status);
    }
}

void ICUBackwardSearch(UStringSearch *srch, const char16_t* source, int32_t sourceLen, const char16_t* pattern, int32_t patternLen, UErrorCode* status) {
    int32_t match;
    
    match = usearch_last(srch, status);
    while (match != USEARCH_DONE) {
        match = usearch_previous(srch, status);
    }
}

#endif /* _STRSRCHPERF_H */
