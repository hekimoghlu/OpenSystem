/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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


#include <jni.h>

#include <stddef.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>

#include <android/log.h>

#include "testenv.h"

#define TAG "CommonCrypto"

static int pipeFDs[2];
static pthread_t stdio_redirect_thread;

static void *stdio_redirect_thread_func(void *context) {
    ssize_t bytesRead;

    char readBuf[4000];
    while ((bytesRead = read(pipeFDs[0], readBuf, sizeof readBuf - 1)) > 0) {

        if (readBuf[bytesRead - 1] == '\n') {
            --bytesRead;
        }

        readBuf[bytesRead] = 0;

        __android_log_write(ANDROID_LOG_DEBUG, TAG, readBuf);
    }

    return 0;
}

static int start_stdio_redirect_thread() {
    setvbuf(stdout, 0, _IOLBF, 0);
    setvbuf(stderr, 0, _IONBF, 0);

    pipe(pipeFDs);
    dup2(pipeFDs[1], 1);
    dup2(pipeFDs[1], 2);

    if (pthread_create(&stdio_redirect_thread, 0, stdio_redirect_thread_func, 0) == -1) {
        return -1;
    }

    pthread_detach(stdio_redirect_thread);

    return 0;
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved) {
    if (start_stdio_redirect_thread() != 0) {
        return JNI_ERR;
    }

    return JNI_VERSION_1_6;
}


JNIEXPORT jint JNICALL Java_com_apple_commoncrypto_CommonCryptoTestShim_startTest
        (JNIEnv *env, jobject obj) {
    return tests_begin(0, NULL);
}

