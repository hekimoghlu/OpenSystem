/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <sys/time.h>
#include <time.h>
#include <CoreFoundation/CoreFoundation.h>
#include "mylog.h"
#include "symbol_scope.h"

/**
 ** eapolclient logging
 **/

STATIC uint32_t	S_log_flags;

PRIVATE_EXTERN uint32_t
eapolclient_log_flags(void)
{
    return (S_log_flags);
}

STATIC void
enable_default_logging(void)
{
    /* enable default logging */
    S_log_flags
	= kLogFlagBasic
	| kLogFlagConfig
	| kLogFlagTunables
	| kLogFlagDisableInnerDetails;
    return;
}

PRIVATE_EXTERN void
eapolclient_log_set_flags(uint32_t log_flags, bool log_it)
{
    if (S_log_flags == log_flags) {
	if (log_flags == 0) {
	    enable_default_logging();
	}
	return;
    }
    if (log_flags != 0) {
	if (log_it) {
	    EAPLOG(LOG_NOTICE, "Verbose mode enabled (LogFlags = 0x%x)",
		   log_flags);
	}
    }
    else {
	if (log_it) {
	    EAPLOG(LOG_NOTICE, "Verbose mode disabled");
	}
    }
    if (log_flags != 0) {
	S_log_flags = log_flags;
    }
    else {
	enable_default_logging();
    }
    return;
}

PRIVATE_EXTERN bool
eapolclient_should_log(uint32_t flags)
{
    return ((S_log_flags & flags) != 0);
}
