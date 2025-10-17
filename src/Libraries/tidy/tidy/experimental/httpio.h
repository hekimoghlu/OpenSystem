/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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

#ifndef __HTTPIO_H__
#define __HTTPIO_H__

#include "platform.h"
#include "tidy.h"

#ifdef WIN32
# include <winsock.h>
# define ECONNREFUSED WSAECONNREFUSED
#else
# include <sys/socket.h>
# include <netdb.h>
# include <netinet/in.h>
#ifndef __BEOS__
# include <arpa/inet.h>
#endif
#endif /* WIN32 */

TIDY_STRUCT
typedef struct _HTTPInputSource
{
    TidyInputSource tis;    //  This declaration must be first and must not be changed!

    tmbstr pHostName;
    tmbstr pResource;
    unsigned short nPort, nextBytePos, nextUnGotBytePos, nBufSize;
    SOCKET s;
    char buffer[1024];
    char unGetBuffer[16];

} HTTPInputSource;

/*  get next byte from input source */
int HTTPGetByte( HTTPInputSource *source );

/*  unget byte back to input source */
void HTTPUngetByte( HTTPInputSource *source, uint byteValue );

/* check if input source at end */
Bool HTTPIsEOF( HTTPInputSource *source );

int parseURL( HTTPInputSource* source, tmbstr pUrl );

int openURL( HTTPInputSource* source, tmbstr pUrl );

void closeURL( HTTPInputSource *source );

#endif