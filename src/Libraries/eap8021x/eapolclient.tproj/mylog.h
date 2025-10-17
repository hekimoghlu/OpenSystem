/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 30, 2023.
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
 * November 8, 2001	Dieter Siegmund
 * - created
 */

#ifndef _S_MYLOG_H
#define _S_MYLOG_H

#include <stdio.h>
#include <sys/types.h>
#include <stdbool.h>
#include "EAPLog.h"

/**
 ** eapolclient logging
 **/
enum {
    kLogFlagBasic 		= 0x00000001,
    kLogFlagConfig		= 0x00000002,
    kLogFlagStatusDetails	= 0x00000004,
    kLogFlagTunables		= 0x00000008,
    kLogFlagPacketDetails 	= 0x00000010,
    kLogFlagDisableInnerDetails	= 0x00001000, /* e.g. LogFlags 0xffffefff */

};

uint32_t
eapolclient_log_flags(void);

void
eapolclient_log_set_flags(uint32_t log_flags, bool log_it);

bool
eapolclient_should_log(uint32_t flags);

#define eapolclient_log(__flags, __format, ...)			\
    if (eapolclient_should_log(__flags))			\
	EAPLOG(LOG_INFO, __format, ## __VA_ARGS__)

#endif /* _S_MYLOG_H */

