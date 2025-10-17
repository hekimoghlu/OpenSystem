/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 27, 2023.
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
#ifndef NAMED_MAIN_H
#define NAMED_MAIN_H 1

/*! \file */

#ifdef ISC_MAIN_HOOK
#define main(argc, argv) bindmain(argc, argv)
#endif

/*
 * Commandline arguments for named; also referenced in win32/ntservice.c
 */
#define NS_MAIN_ARGS "46c:C:d:D:E:fFgi:lM:m:n:N:p:P:sS:t:T:U:u:vVx:"

ISC_PLATFORM_NORETURN_PRE void
ns_main_earlyfatal(const char *format, ...)
ISC_FORMAT_PRINTF(1, 2) ISC_PLATFORM_NORETURN_POST;

void
ns_main_earlywarning(const char *format, ...) ISC_FORMAT_PRINTF(1, 2);

void
ns_main_setmemstats(const char *);

#endif /* NAMED_MAIN_H */
