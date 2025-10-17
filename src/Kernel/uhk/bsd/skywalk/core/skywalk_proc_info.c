/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#include <sys/types.h>
#include <sys/kernel_types.h>
#include <sys/errno.h>
#include <sys/kernel.h>
#include <sys/file_internal.h>
#include <sys/proc_info.h>
#include <sys/sys_domain.h>
#include <sys/kern_event.h>
#include <string.h>
#include <skywalk/os_skywalk_private.h>

static uint32_t
ch_mode_to_flags(uint32_t ch_mode)
{
	uint32_t        flags = 0;

	if ((ch_mode & CHMODE_MONITOR_RX) != 0) {
		flags |= PROC_CHANNEL_FLAGS_MONITOR_RX;
	}
	if ((ch_mode & CHMODE_MONITOR_TX) != 0) {
		flags |= PROC_CHANNEL_FLAGS_MONITOR_TX;
	}
	if ((ch_mode & CHMODE_MONITOR_NO_COPY) != 0) {
		flags |= PROC_CHANNEL_FLAGS_MONITOR_NO_COPY;
	}
	if ((ch_mode & CHMODE_EXCLUSIVE) != 0) {
		flags |= PROC_CHANNEL_FLAGS_EXCLUSIVE;
	}
	if ((ch_mode & CHMODE_USER_PACKET_POOL) != 0) {
		flags |= PROC_CHANNEL_FLAGS_USER_PACKET_POOL;
	}
	if ((ch_mode & CHMODE_DEFUNCT_OK) != 0) {
		flags |= PROC_CHANNEL_FLAGS_DEFUNCT_OK;
	}
	if ((ch_mode & CHMODE_LOW_LATENCY) != 0) {
		flags |= PROC_CHANNEL_FLAGS_LOW_LATENCY;
	}
	return flags;
}

errno_t
fill_channelinfo(struct kern_channel *channel, struct proc_channel_info *info)
{
	errno_t                 error = 0;
	struct kern_nexus       *nexus;

	lck_mtx_lock(&channel->ch_lock);
	uuid_copy(info->chi_instance, channel->ch_info->cinfo_nx_uuid);
	info->chi_port = channel->ch_info->cinfo_nx_port;
	info->chi_flags = ch_mode_to_flags(channel->ch_info->cinfo_ch_mode);
	nexus = channel->ch_nexus;
	if (nexus != NULL) {
		struct kern_nexus_provider *nx_prov = nexus->nx_prov;
		if (nx_prov != NULL) {
			struct kern_nexus_domain_provider *dom_prov;
			dom_prov = nx_prov->nxprov_dom_prov;
			if (dom_prov != NULL) {
				struct nxdom *dom;
				dom = dom_prov->nxdom_prov_dom;
				if (dom != NULL) {
					info->chi_type = dom->nxdom_type;
				}
			}
		}
	}
	lck_mtx_unlock(&channel->ch_lock);
	return error;
}
