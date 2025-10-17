/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/********************************************************************
 * COPYRIGHT: 
 * Copyright (c) 1999-2015, International Business Machines Corporation and
 * others. All Rights Reserved.
 ********************************************************************/


#include "simplethread.h"

#include <thread>
#include "unicode/utypes.h"
#include "intltest.h"


SimpleThread::SimpleThread() {
}

SimpleThread::~SimpleThread() {
    this->join();     // Avoid crashes if user neglected to join().
}

int SimpleThread::start() {
    fThread = std::thread(&SimpleThread::run, this);
    return fThread.joinable() ? 0 : 1;
}

void SimpleThread::join() {
    if (fThread.joinable()) {
        fThread.join();
    }
}



class ThreadPoolThread: public SimpleThread {
  public:
    ThreadPoolThread(ThreadPoolBase *pool, int32_t threadNum) : fPool(pool), fNum(threadNum) {}
    virtual void run() override { fPool->callFn(fNum); }
    ThreadPoolBase *fPool;
    int32_t         fNum;
};


ThreadPoolBase::ThreadPoolBase(IntlTest *test, int32_t howMany) :
        fIntlTest(test), fNumThreads(howMany), fThreads(nullptr) {
    fThreads = new SimpleThread *[fNumThreads];
    if (fThreads == nullptr) {
        fIntlTest->errln("%s:%d memory allocation failure.", __FILE__, __LINE__);
        return;
    }

    for (int i=0; i<fNumThreads; i++) {
        fThreads[i] = new ThreadPoolThread(this, i);
        if (fThreads[i] == nullptr) {
            fIntlTest->errln("%s:%d memory allocation failure.", __FILE__, __LINE__);
        }
    }
}

void ThreadPoolBase::start() {
    for (int i=0; i<fNumThreads; i++) {
        if (fThreads && fThreads[i]) {
            fThreads[i]->start();
        }
    }
}

void ThreadPoolBase::join() {
    for (int i=0; i<fNumThreads; i++) {
        if (fThreads && fThreads[i]) {
            fThreads[i]->join();
        }
    }
}

ThreadPoolBase::~ThreadPoolBase() {
    if (fThreads) {
        for (int i=0; i<fNumThreads; i++) {
            delete fThreads[i];
            fThreads[i] = nullptr;
        }
        delete[] fThreads;
        fThreads = nullptr;
    }
}
