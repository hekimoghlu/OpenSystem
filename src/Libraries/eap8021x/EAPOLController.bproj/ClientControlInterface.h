/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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

#ifndef _S_CLIENTCONTROLINTERFACE
#define _S_CLIENTCONTROLINTERFACE

#include <stdint.h>

#define kEAPOLClientControlCommand		CFSTR("Command")
#define kEAPOLClientControlConfiguration	CFSTR("Configuration")
#define kEAPOLClientControlUserInput		CFSTR("UserInput")
#define kEAPOLClientControlMode			CFSTR("Mode") /* CFNumber(EAPOLControlMode) */
#define kEAPOLClientControlPacketIdentifier	CFSTR("PacketIdentifier") /* CFNumber */

enum {
    kEAPOLClientControlCommandRun = 1,
    kEAPOLClientControlCommandStop = 2,
    kEAPOLClientControlCommandRetry = 3,
    kEAPOLClientControlCommandTakeUserInput = 4,
};
typedef uint32_t EAPOLClientControlCommand;

#endif /* _S_CLIENTCONTROLINTERFACE */
