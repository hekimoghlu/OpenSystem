/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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
/*
 * Modification History
 *
 * July 9, 2001			Allan Nathanson <ajn@apple.com>
 * - added "-r" option for checking network reachability
 * - added "-w" option to check/wait for the presence of a
 *   dynamic store key.
 *
 * November 9, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _TESTS_H
#define _TESTS_H

#include <sys/cdefs.h>

__BEGIN_DECLS

void	do_checkReachability		(int argc, char * const argv[]);
void	do_watchReachability		(int argc, char * const argv[]);
void	do_renew			(char *interface);
void	do_showDNSConfiguration		(int argc, char * const argv[]);
void	do_watchDNSConfiguration	(int argc, char * const argv[]);
void	do_showProxyConfiguration	(int argc, char * const argv[]);
void	do_snapshot			(int argc, char * const argv[]);
void	do_wait				(char *waitKey, int timeout);
void	do_showNWI			(int argc, char * const argv[]);
void	do_watchNWI			(int argc, char * const argv[]);
void	do_advisory			(const char * interface, Boolean watch, int argc, char * const argv[]);
void	do_rank				(const char * interface, Boolean watch, int argc, char * const argv[]);

__END_DECLS

#endif /* !_TESTS_H */
