/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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
#include<config.h>

#include <roken.h>

#ifndef _WIN32
#error Only implemented for Windows
#endif

volatile LONG _startup_count = 0;

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rk_WSAStartup(void)
{
    WSADATA wsad;

    if (!WSAStartup( MAKEWORD(2, 2), &wsad )) {
	if (wsad.wVersion != MAKEWORD(2, 2)) {
	    /* huh? We can't use 2.2? */
	    WSACleanup();
	    return -1;
	}

	InterlockedIncrement(&_startup_count);
	return 0;
    }

    return -1;
}


ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rk_WSACleanup(void)
{
    LONG l;

    if ((l = InterlockedDecrement(&_startup_count)) < 0) {
	l = InterlockedIncrement(&_startup_count) - 1;
    }

    if (l >= 0) {
	return WSACleanup();
    }
    return -1;
}
