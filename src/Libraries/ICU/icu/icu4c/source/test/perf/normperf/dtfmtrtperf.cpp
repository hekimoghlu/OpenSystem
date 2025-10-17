/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
#include "dtfmtrtperf.h"
#include "uoptions.h"
#include <stdio.h>

#include <iostream>
using namespace std;

DateTimeRoundTripPerfTest::DateTimeRoundTripPerfTest(int32_t argc, const char* argv[], UErrorCode& status)
: UPerfTest(argc,argv,status) { }

DateTimeRoundTripPerfTest::~DateTimeRoundTripPerfTest() { }

UPerfFunction* DateTimeRoundTripPerfTest::runIndexedTest(int32_t index, UBool exec,const char* &name, char* par) {

    switch (index) 
    {
        TESTCASE(0,RoundTripLocale1);     // 1 locale
        TESTCASE(1,RoundTripLocale10);    // 10 locales  
        TESTCASE(2,RoundTripLocale11);    // 11 locales
        TESTCASE(3,RoundTripLocale21);    // 21 locales w/ reverse order
        default: 
            name = ""; 
            return nullptr;
    }
    return nullptr;

}

UPerfFunction* DateTimeRoundTripPerfTest::RoundTripLocale1(){
    DateTimeRoundTripFunction* func= new DateTimeRoundTripFunction(1);
    return func;
}

UPerfFunction* DateTimeRoundTripPerfTest::RoundTripLocale10(){
    DateTimeRoundTripFunction* func= new DateTimeRoundTripFunction(10);
    return func;
}

UPerfFunction* DateTimeRoundTripPerfTest::RoundTripLocale11(){
    DateTimeRoundTripFunction* func= new DateTimeRoundTripFunction(11);
    return func;
}

UPerfFunction* DateTimeRoundTripPerfTest::RoundTripLocale21(){
    DateTimeRoundTripFunction* func= new DateTimeRoundTripFunction(21);
    return func;
}

int main(int argc, const char* argv[]){

	cout << "ICU version - " << U_ICU_VERSION << endl;

    UErrorCode status = U_ZERO_ERROR;
    DateTimeRoundTripPerfTest test(argc, argv, status);
    if(U_FAILURE(status)){
		cout << "initialization failed! " << status << endl;
        return status;
    }

    if(test.run()==false){
		cout << "run failed!" << endl;
        fprintf(stderr,"FAILED: Tests could not be run please check the arguments.\n");
        return -1;
    }

	cout << "done!" << endl;
    return 0;
}
