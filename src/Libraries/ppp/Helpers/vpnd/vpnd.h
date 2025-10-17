/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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
#include <sys/syslog.h>


/*
 * Procedures exported from sys-*.c
 */
int  ppp_available __P((void));		/* Test whether ppp kernel support exists */
CFStringRef CopyDefaultIPAddress(void); 	/* Copy the IPAddress of the default interface */
int get_route_interface(struct sockaddr *src, const struct sockaddr *dst, char *if_name); /* get the interface for a given address */
int find_address(const struct sockaddr_in *address, char *interface); /* check if an interface has a given address */


/*
 * Exit status values.
 */
#define EXIT_OK			0
#define EXIT_FATAL_ERROR	1
#define EXIT_OPTION_ERROR	2
#define EXIT_NOT_ROOT		3
#define EXIT_NO_KERNEL_SUPPORT	4
#define EXIT_USER_REQUEST	5


#define	PLUGINS_DIR 	"/System/Library/SystemConfiguration/PPPController.bundle/Contents/PlugIns/"

void vpnlog(int nSyslogPriority, char *format_str, ...)  __printflike(2,3);
int update_prefs(void);
void toggle_debug(void);
void set_terminate(void);

int readn(int ref, void *data, int len);
int writen(int ref, void *data, int len);
