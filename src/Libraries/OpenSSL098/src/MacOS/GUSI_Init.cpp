/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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
#define GUSI_SOURCE
#include <GUSIConfig.h>
#include <sys/cdefs.h>

/* Declarations of Socket Factories */

__BEGIN_DECLS
void GUSIwithInetSockets();
void GUSIwithLocalSockets();
void GUSIwithMTInetSockets();
void GUSIwithMTTcpSockets();
void GUSIwithMTUdpSockets();
void GUSIwithOTInetSockets();
void GUSIwithOTTcpSockets();
void GUSIwithOTUdpSockets();
void GUSIwithPPCSockets();
void GUSISetupFactories();
__END_DECLS

/* Configure Socket Factories */

void GUSISetupFactories()
{
#ifdef GUSISetupFactories_BeginHook
	GUSISetupFactories_BeginHook
#endif
	GUSIwithInetSockets();
#ifdef GUSISetupFactories_EndHook
	GUSISetupFactories_EndHook
#endif
}

/* Declarations of File Devices */

__BEGIN_DECLS
void GUSIwithDConSockets();
void GUSIwithNullSockets();
void GUSISetupDevices();
__END_DECLS

/* Configure File Devices */

void GUSISetupDevices()
{
#ifdef GUSISetupDevices_BeginHook
	GUSISetupDevices_BeginHook
#endif
	GUSIwithNullSockets();
#ifdef GUSISetupDevices_EndHook
	GUSISetupDevices_EndHook
#endif
}

/**************** END GUSI CONFIGURATION *************************/
