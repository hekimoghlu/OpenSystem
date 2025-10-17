/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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
#ifndef CONFIG_AGENT_INFO_H
#define CONFIG_AGENT_INFO_H

#include <dnsinfo.h>
#include <net/network_agent.h>
#include <xpc/private.h>

__BEGIN_DECLS

#define kConfigAgentDomain                      "SystemConfig"

#define kConfigAgentTypeProxy                   "ProxyAgent"
#define kConfigAgentTypeDNS                     "DNSAgent"

/*
	Returns true for agent with type DNSAgent and domain SystemConfig
 */
boolean_t
is_config_agent_type_dns		(const struct netagent *agent)		API_AVAILABLE(macos(10.12), ios(10.0));

/*
	Returns true for agent with type ProxyAgent and domain SystemConfig
 */
boolean_t
is_config_agent_type_proxy		(const struct netagent *agent)		API_AVAILABLE(macos(10.12), ios(10.0));

/*
	Returns xpc_object_t corresponding to the raw DNSAgent data
		NULL if the agent is NOT a DNSAgent
 */
xpc_object_t
config_agent_copy_dns_information	(const struct netagent	*agentStruct)	API_AVAILABLE(macos(10.12), ios(10.0));

/*
	Returns xpc_object_t (XPC_TYPE_ARRAY) corresponding to the DNS nameservers
		NULL if the agent is NOT a DNSAgent
 */
xpc_object_t
config_agent_get_dns_nameservers	(xpc_object_t resolver)			API_AVAILABLE(macos(10.12), ios(10.0));

/*
	Returns xpc_object_t (XPC_TYPE_ARRAY) corresponding to the DNS search domains
		NULL if the agent is NOT a DNSAgent
 */
xpc_object_t
config_agent_get_dns_searchdomains	(xpc_object_t resolver)			API_AVAILABLE(macos(10.12), ios(10.0));

/*
	Frees the xpc_object_t returned by config_agent_copy_dns_information()
 */
void
config_agent_free_dns_information	(xpc_object_t resolver)			API_AVAILABLE(macos(10.12), ios(10.0));

/*
	Returns xpc_object_t corresponding to the raw ProxyAgent data
		NULL if the agent is NOT a ProxyAgent
 */
xpc_object_t
config_agent_copy_proxy_information	(const struct netagent	*agentStruct)	API_AVAILABLE(macos(10.12), ios(10.0));

/*
	Updates the proxy config with PAC, if applicable. The proxyConfig MUST be
	of type XPC_TYPE_ARRAY containing a XPC_TYPE_DICTIONARY. This format is
	returned by config_agent_copy_proxy_information()

	Returns xpc_object_t to be freed by the caller.
		NULL if the the provided configuration does not need any update.
 */
xpc_object_t
config_agent_update_proxy_information	(xpc_object_t proxyConfig)		API_AVAILABLE(macos(10.12), ios(10.0));

/*
	Frees the xpc_object_t returned by config_agent_copy_proxy_information()
 */
void
config_agent_free_proxy_information	(xpc_object_t proxyConfig)		API_AVAILABLE(macos(10.12), ios(10.0));

__END_DECLS

#endif	/* CONFIG_AGENT_INFO_H */
