/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#ifndef _S_CONTROLLER_H
#define _S_CONTROLLER_H

#include <sys/types.h>
#include <sys/queue.h>
#include <mach/mach.h>
#include <CoreFoundation/CFDictionary.h>
#include <CoreFoundation/CFString.h>
#include <TargetConditionals.h>

#include "eapolcontroller_types.h"

int 
ControllerCopyStateAndStatus(if_name_t if_name, 
			     int * state,
			     CFDictionaryRef * status_dict);

int 
ControllerGetState(if_name_t if_name, int * state);

int
ControllerStart(if_name_t if_name, uid_t uid, gid_t gid,
		CFDictionaryRef config_dict, mach_port_t bootstrap,
		mach_port_t au_session);

int
ControllerStartSystem(if_name_t if_name, uid_t uid, gid_t gid,
		      CFDictionaryRef options);

int
ControllerUpdate(if_name_t if_name, uid_t uid, gid_t gid,
		 CFDictionaryRef config_dict);

int
ControllerProvideUserInput(if_name_t if_name, uid_t uid, gid_t gid,
			   CFDictionaryRef user_input_dict);

int
ControllerRetry(if_name_t if_name, uid_t uid, gid_t gid);

int
ControllerStop(if_name_t if_name, uid_t uid, gid_t gid);

#if ! TARGET_OS_IPHONE

int 
ControllerCopyLoginWindowConfiguration(if_name_t if_name, 
				       CFDictionaryRef * config_data_p);
int
ControllerCopyAutoDetectInformation(CFDictionaryRef * info_p);

boolean_t
ControllerDidUserCancel(if_name_t if_name);

#endif /* ! TARGET_OS_IPHONE */

int
ControllerClientAttach(pid_t pid, if_name_t if_name,
		       mach_port_t notify_port,
		       mach_port_t * session_port,
		       CFDictionaryRef * control_dict);

void
ControllerClientGetSession(pid_t pid, if_name_t if_name,
			   mach_port_t * bootstrap,
			   mach_port_t * au_session);

int
ControllerClientDetach(mach_port_t session_port);

int
ControllerClientGetConfig(mach_port_t session_port,
			  CFDictionaryRef * control_dict);

int
ControllerClientReportStatus(mach_port_t session_port,
			     CFDictionaryRef status_dict);

int
ControllerClientForceRenew(mach_port_t session_port);

int
ControllerClientPortDead(mach_port_t session_port);

#if ! TARGET_OS_IPHONE

int
ControllerClientUserCancelled(mach_port_t session_port);

#endif /* ! TARGET_OS_IPHONE */

#endif /* _S_CONTROLLER_H */
