/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#import "controller.h"

static Boolean
haveNetworkExtensionFramework(void)
{
	Boolean	haveFramework;

	haveFramework = ([NEPolicy class] != nil);
	return haveFramework;
}

void
process_AgentMonitor(void)
{
	if (!haveNetworkExtensionFramework()) {
		return;
	}

	SC_log(LOG_DEBUG, "Triggering AgentMonitor");
	@autoreleasepool {
		AgentController *controller = [AgentController sharedController];
		if (controller == nil) {
			SC_log(LOG_ERR, "AgentController could not be initialized");
			return;
		}

		dispatch_sync(controller.controllerQueue, ^{
			[[AgentController sharedController] processDNSChanges];
			[[AgentController sharedController] processProxyChanges];
		});
	}

	return;
}

void
process_AgentMonitor_DNS(void)
{
	if (!haveNetworkExtensionFramework()) {
		return;
	}

	SC_log(LOG_DEBUG, "Triggering AgentMonitor for DNS");
	@autoreleasepool {
		AgentController *controller = [AgentController sharedController];
		if (controller == nil) {
			SC_log(LOG_ERR, "AgentController could not be initialized");
			return;
		}

		dispatch_sync(controller.controllerQueue, ^{
			[[AgentController sharedController] processDNSChanges];
		});
	}

	return;
}

void
process_AgentMonitor_Proxy(void)
{
	if (!haveNetworkExtensionFramework()) {
		return;
	}

	SC_log(LOG_DEBUG, "Triggering AgentMonitor for Proxy");
	@autoreleasepool {
		AgentController *controller = [AgentController sharedController];
		if (controller == nil) {
			SC_log(LOG_ERR, "AgentController could not be initialized");
			return;
		}

		dispatch_sync(controller.controllerQueue, ^{
			[[AgentController sharedController] processProxyChanges];
		});
	}

	return;
}

const void *
copy_proxy_information_for_agent_uuid(uuid_t uuid, uint64_t *length)
{
	__block const void *buffer = NULL;

	if (!haveNetworkExtensionFramework()) {
		return NULL;
	}

	@autoreleasepool {
		AgentController *controller = [AgentController sharedController];
		if (controller == nil) {
			SC_log(LOG_ERR, "AgentController could not be initialized");
			return NULL;
		}

		dispatch_sync(controller.controllerQueue, ^{
			buffer = [[AgentController sharedController] copyProxyAgentData:uuid
										 length:length];
		});
	}

	return buffer;
}

const void *
copy_dns_information_for_agent_uuid(uuid_t uuid, uint64_t *length)
{
	__block const void *buffer = NULL;

	if (!haveNetworkExtensionFramework()) {
		return NULL;
	}

	@autoreleasepool {
		AgentController *controller = [AgentController sharedController];
		if (controller == nil) {
			SC_log(LOG_ERR, "AgentController could not be initialized");
			return NULL;
		}

		dispatch_sync(controller.controllerQueue, ^{
			buffer = [[AgentController sharedController] copyDNSAgentData:uuid
									       length:length];
		});
	}

	return buffer;
}
