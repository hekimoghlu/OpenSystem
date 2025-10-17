/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 1, 2022.
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
#include <algorithm>
#include <vector>
#include <string>

#include "unicode/locid.h"
#include "unicode/uperf.h"

//
// Test case ...
//
class LocaleCreateCanonical : public UPerfFunction {
public:
    LocaleCreateCanonical() {
        testCases.emplace_back("en");
        testCases.emplace_back("en-US");
        testCases.emplace_back("ja-JP");
        testCases.emplace_back("zh-Hant-CN");
        testCases.emplace_back("hy-SU");
    }
    ~LocaleCreateCanonical() {  }
    void call(UErrorCode* /*status*/) override
    {
        std::for_each(testCases.begin(), testCases.end(),
                      [](const std::string& s)
                      {
                          Locale l = Locale::createCanonical(s.c_str());
                      });
    }
    long getOperationsPerIteration() override { return testCases.size(); }
    long getEventsPerIteration() override { return testCases.size(); }
private:
    std::vector<std::string> testCases;
};

class LocaleCanonicalizationPerfTest : public UPerfTest
{
public:
    LocaleCanonicalizationPerfTest(
        int32_t argc, const char *argv[], UErrorCode &status)
            : UPerfTest(argc, argv, nullptr, 0, "localecanperf", status)
    {
    }

    ~LocaleCanonicalizationPerfTest()
    {
    }
    UPerfFunction* runIndexedTest(
        int32_t index, UBool exec, const char*& name, char* par = nullptr) override;

private:
    UPerfFunction* TestLocaleCreateCanonical()
    {
        return new LocaleCreateCanonical();
    }
};

UPerfFunction*
LocaleCanonicalizationPerfTest::runIndexedTest(
    int32_t index, UBool exec, const char *&name, char *par /*= nullptr*/)
{
    (void)par;
    TESTCASE_AUTO_BEGIN;

    TESTCASE_AUTO(TestLocaleCreateCanonical);

    TESTCASE_AUTO_END;
    return nullptr;
}

int main(int argc, const char *argv[])
{
    UErrorCode status = U_ZERO_ERROR;
    LocaleCanonicalizationPerfTest test(argc, argv, status);

    if (U_FAILURE(status)){
        fprintf(stderr, "The error is %s\n", u_errorName(status));
        test.usage();
        return status;
    }

    if (test.run() == false){
        test.usage();
        fprintf(stderr, "FAILED: Tests could not be run please check the arguments.\n");
        return -1;
    }
    return 0;
}
