/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#define VERSION "0.4.0"

#include "config.h"

/* Probably too many inclusions but this is to keep 'gcc -Wall' happy... */
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <arpa/nameser.h>
#ifdef HAVE_ARPA_NAMESER_COMPAT_H
#include <arpa/nameser_compat.h>
#endif
#include <resolv.h>

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif

#if SIZEOF_LONG == 4
#define u_int32_t unsigned long
#ifndef int32_t 
#define int32_t   long
#endif
#else
#define u_int32_t unsigned int
#ifndef int32_t 
#define int32_t   int
#endif
#endif

#if SIZEOF_CHAR == 1
#define u_int8_t unsigned char
#ifndef int8_t 
#define int8_t   char
#endif
#else 
#if SIZEOF_SHORT == 1
#define u_int8_t unsigned short
#ifndef int8_t 
#define int8_t   short
#endif
#else
#error "No suitable native type for storing bytes"
#endif
#endif

#ifndef INADDR_NONE
#define INADDR_NONE (in_addr_t)-1
#endif

struct list_in_addr
  {
    struct in_addr addr;
    void *next;
  };

void usage ();
void panic ();

char *getlocbyname ();
char *getlocbyaddr ();
char *getlocbynet ();
char *findRR ();
struct list_in_addr *findA ();

extern char *progname;
extern short debug;
