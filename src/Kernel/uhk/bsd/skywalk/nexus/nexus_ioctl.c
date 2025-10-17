/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
#include <skywalk/os_skywalk_private.h>
#include <IOKit/IOBSD.h>

static int
nxioctl_check_entitlement(u_long cmd)
{
	boolean_t entitled = FALSE;

	if (kauth_cred_issuser(kauth_cred_get())) {
		return 0;
	}
	switch (cmd) {
	case NXIOC_ADD_TRAFFIC_RULE_INET:
	case NXIOC_REMOVE_TRAFFIC_RULE:
		entitled = IOCurrentTaskHasEntitlement(
			NXCTL_TRAFFIC_RULE_WRITE_ENTITLEMENT);
		break;
	case NXIOC_GET_TRAFFIC_RULES:
		entitled = IOCurrentTaskHasEntitlement(
			NXCTL_TRAFFIC_RULE_READ_ENTITLEMENT);
		break;
	default:
		SK_ERR("invalid command %x", cmd);
		return ENOTSUP;
	}
	return entitled ? 0 : EPERM;
}

int
nxioctl(struct nxctl *nxctl, u_long cmd, caddr_t data, proc_t procp)
{
	int err;

	if ((err = nxioctl_check_entitlement(cmd)) != 0) {
		return err;
	}
	switch (cmd) {
	case NXIOC_ADD_TRAFFIC_RULE_INET:
		return nxioctl_add_traffic_rule_inet(nxctl, data, procp);
	case NXIOC_REMOVE_TRAFFIC_RULE:
		return nxioctl_remove_traffic_rule(nxctl, data, procp);
	case NXIOC_GET_TRAFFIC_RULES:
		return nxioctl_get_traffic_rules(nxctl, data, procp);
	default:
		SK_ERR("invalid command %x", cmd);
		return ENOTSUP;
	}
}
