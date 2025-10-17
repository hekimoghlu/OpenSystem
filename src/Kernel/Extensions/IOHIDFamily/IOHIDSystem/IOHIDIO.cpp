/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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
/* 	Copyright (c) 1992 NeXT Computer, Inc.  All rights reserved. 
 *
 * EventIO.m - Event System MiG interface for driver control and status.
 *
 * HISTORY
 * 2-April-92    Mike Paquette at NeXT 
 *      Created. 
 */

#include <IOKit/system.h>

#include <IOKit/hidsystem/IOHIDTypes.h>
#include <IOKit/hidsystem/IOHIDSystem.h>
#include <IOKit/hidsystem/IOHIDShared.h>

#include <IOKit/hidsystem/ev_private.h>	/* Per-machine configuration info */

/*
 * Additional kernel API to drivers using the Event Driver
 */
int EventCoalesceDisplayCmd( int cmd, int oldcmd );
int EventCoalesceDisplayCmd( int cmd, int oldcmd )
{
	static const char coalesce[4][4] = {
	    /* nop */  {EVNOP,  EVHIDE, EVSHOW, EVMOVE},
	    /* hide */ {EVHIDE, EVHIDE, EVNOP,  EVSHOW},
	    /* show */ {EVSHOW, EVNOP,  EVSHOW, EVSHOW},
	    /* move */ {EVMOVE, EVHIDE, EVSHOW, EVMOVE}
	};
	if ( cmd < EVLEVEL )	// coalesce EVNOP thru EVMOVE only
	    cmd = coalesce[oldcmd & 3][cmd & 3];
	return cmd;
}

