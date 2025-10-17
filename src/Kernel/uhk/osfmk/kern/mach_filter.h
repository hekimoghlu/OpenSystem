/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
#ifndef _KERN_MACH_FILTER_H_
#define _KERN_MACH_FILTER_H_

#if KERNEL_PRIVATE

#include <sys/cdefs.h>
#include <mach/message.h>
#include <mach/port.h>

/* Sandbox-specific calls for task based message filtering */
typedef boolean_t (*mach_msg_fetch_filter_policy_cbfunc_t) (struct task *task, void *portlabel,
    mach_msg_id_t msgid, mach_msg_filter_id *fpid);

typedef kern_return_t (*mach_msg_filter_alloc_service_port_sblabel_cbfunc_t) (mach_service_port_info_t service_port_info,
    void **sblabel);

typedef void (*mach_msg_filter_dealloc_service_port_sblabel_cbfunc_t) (void *sblabel);

typedef void* (*mach_msg_filter_derive_sblabel_from_service_port_cbfunc_t) (void *service_port_sblabel,
    boolean_t *send_side_filtering);

typedef kern_return_t (*mach_msg_filter_get_connection_port_filter_policy_cbfunc_t) (void *service_port_sblabel,
    void *connection_port_sblabel, uint64_t *fpid);

/* Will be called with the port lock held */
typedef void (*mach_msg_filter_retain_sblabel_cbfunc_t) (void * sblabel);

struct mach_msg_filter_callbacks {
	unsigned int version;
	/* v0 */
	const mach_msg_fetch_filter_policy_cbfunc_t fetch_filter_policy;

	/* v1 */
	const mach_msg_filter_alloc_service_port_sblabel_cbfunc_t alloc_service_port_sblabel;
	const mach_msg_filter_dealloc_service_port_sblabel_cbfunc_t dealloc_service_port_sblabel;
	const mach_msg_filter_derive_sblabel_from_service_port_cbfunc_t derive_sblabel_from_service_port;
	const mach_msg_filter_get_connection_port_filter_policy_cbfunc_t get_connection_port_filter_policy;
	const mach_msg_filter_retain_sblabel_cbfunc_t retain_sblabel;
};

#define MACH_MSG_FILTER_CALLBACKS_VERSION_0 (0) /* up-to fetch_filter_policy */
#define MACH_MSG_FILTER_CALLBACKS_VERSION_1 (1) /* up-to derive_sblabel_from_service_port */
#define MACH_MSG_FILTER_CALLBACKS_CURRENT MACH_MSG_FILTER_CALLBACKS_VERSION_1

__BEGIN_DECLS

int mach_msg_filter_register_callback(const struct mach_msg_filter_callbacks *callbacks);

__END_DECLS

#endif /* KERNEL_PRIVATE */

#if XNU_KERNEL_PRIVATE
extern struct mach_msg_filter_callbacks mach_msg_filter_callbacks;

static inline bool __pure2
mach_msg_filter_at_least(unsigned int version)
{
	if (version == 0) {
		/*
		 * a non initialized cb struct looks the same as v0
		 * so we need a null check for that one
		 */
		return mach_msg_filter_callbacks.fetch_filter_policy != NULL;
	}
	return mach_msg_filter_callbacks.version >= version;
}

/* v0 */
#define mach_msg_fetch_filter_policy_callback \
	(mach_msg_filter_callbacks.fetch_filter_policy)

/* v1 */
#define mach_msg_filter_alloc_service_port_sblabel_callback \
	(mach_msg_filter_callbacks.alloc_service_port_sblabel)
#define mach_msg_filter_dealloc_service_port_sblabel_callback \
	(mach_msg_filter_callbacks.dealloc_service_port_sblabel)
#define mach_msg_filter_derive_sblabel_from_service_port_callback \
	(mach_msg_filter_callbacks.derive_sblabel_from_service_port)
#define mach_msg_filter_get_connection_port_filter_policy_callback \
	(mach_msg_filter_callbacks.get_connection_port_filter_policy)
#define mach_msg_filter_retain_sblabel_callback \
	(mach_msg_filter_callbacks.retain_sblabel)

extern
boolean_t mach_msg_fetch_filter_policy(void *portlabel, mach_msg_id_t msgh_id, mach_msg_filter_id *fid);
#endif /* XNU_KERNEL_PRIVATE */

#endif /* _KERN_MACH_FILTER_H_ */
