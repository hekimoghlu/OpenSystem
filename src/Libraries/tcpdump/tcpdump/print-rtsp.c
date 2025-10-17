/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
/* \summary: Real Time Streaming Protocol (RTSP) printer */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "netdissect-stdinc.h"

#include "netdissect.h"

static const char *rtspcmds[] = {
	"DESCRIBE",
	"ANNOUNCE",
	"GET_PARAMETER",
	"OPTIONS",
	"PAUSE",
	"PLAY",
	"RECORD",
	"REDIRECT",
	"SETUP",
	"SET_PARAMETER",
	"TEARDOWN",
	NULL
};

void
rtsp_print(netdissect_options *ndo, const u_char *pptr, u_int len)
{
	ndo->ndo_protocol = "rtsp";
	txtproto_print(ndo, pptr, len, rtspcmds, RESP_CODE_SECOND_TOKEN);
}
