/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 22, 2022.
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
#ifndef _IPC_IPC_SERVICE_PORT_H_
#define _IPC_IPC_SERVICE_PORT_H_

#include <mach/std_types.h>
#include <mach/port.h>
#include <mach/mach_eventlink_types.h>
#include <mach_assert.h>

#include <mach/mach_types.h>
#include <mach/boolean.h>
#include <mach/kern_return.h>

#include <kern/assert.h>
#include <kern/kern_types.h>

#include <ipc/ipc_types.h>
#include <ipc/ipc_object.h>
#include <ipc/ipc_port.h>
#include <kern/waitq.h>
#include <os/refcnt.h>

#ifdef MACH_KERNEL_PRIVATE

__options_decl(ipc_service_port_label_flags_t, uint16_t, {
	ISPL_FLAGS_SPECIAL_PDREQUEST      = 1,/* Special port destroyed notification for service ports */
	ISPL_FLAGS_SEND_PD_NOTIFICATION   = (1 << 1),/* Port destroyed notification is being sent */
	ISPL_FLAGS_BOOTSTRAP_PORT         = (1 << 2),
	ISPL_FLAGS_THROTTLED              = (1 << 3),/* Service throttled by launchd */
});

struct ipc_service_port_label {
	void * XNU_PTRAUTH_SIGNED_PTR("ipc_service_port_label.ispl_sblabel") ispl_sblabel; /* points to the Sandbox's message filtering data structure */
	mach_port_context_t               ispl_launchd_context;     /* context used to guard the port, specific to launchd */
	mach_port_name_t                  ispl_launchd_name;        /* port name in launchd's ipc space */
	ipc_service_port_label_flags_t    ispl_flags;
#if CONFIG_SERVICE_PORT_INFO
	uint8_t             ispl_domain;             /* launchd domain */
	char                *ispl_service_name;       /* string name used to identify the service port */
#endif /* CONFIG_SERVICE_PORT_INFO */
};

typedef struct ipc_service_port_label* ipc_service_port_label_t;

#define IPC_SERVICE_PORT_LABEL_NULL ((ipc_service_port_label_t)NULL)

/*
 * These ispl_flags based macros/functions should be called with the port lock held
 */
#define ipc_service_port_label_is_special_pdrequest(port_splabel) \
    (((port_splabel)->ispl_flags & ISPL_FLAGS_SPECIAL_PDREQUEST) == ISPL_FLAGS_SPECIAL_PDREQUEST)

#define ipc_service_port_label_is_pd_notification(port_splabel) \
    (((port_splabel)->ispl_flags & ISPL_FLAGS_SEND_PD_NOTIFICATION) == ISPL_FLAGS_SEND_PD_NOTIFICATION)

#define ipc_service_port_label_is_bootstrap_port(port_splabel) \
    (((port_splabel)->ispl_flags & ISPL_FLAGS_BOOTSTRAP_PORT) == ISPL_FLAGS_BOOTSTRAP_PORT)

#define ipc_service_port_label_is_throttled(port_splabel) \
	(((port_splabel)->ispl_flags & ISPL_FLAGS_THROTTLED) == ISPL_FLAGS_THROTTLED)

static inline void
ipc_service_port_label_set_flag(ipc_service_port_label_t port_splabel, ipc_service_port_label_flags_t flag)
{
	assert(port_splabel != IPC_SERVICE_PORT_LABEL_NULL);
	port_splabel->ispl_flags |= flag;
}

static inline void
ipc_service_port_label_clear_flag(ipc_service_port_label_t port_splabel, ipc_service_port_label_flags_t flag)
{
	assert(port_splabel != IPC_SERVICE_PORT_LABEL_NULL);
	port_splabel->ispl_flags &= ~flag;
}

/* Function declarations */
kern_return_t
ipc_service_port_label_alloc(mach_service_port_info_t sp_info, void **port_label_ptr);

void
ipc_service_port_label_dealloc(void * ip_splabel, bool service_port);

kern_return_t
ipc_service_port_derive_sblabel(mach_port_name_t service_port_name, void **sblabel_ptr, bool *filter_msgs);

void *
ipc_service_port_get_sblabel(ipc_port_t port);

void
ipc_service_port_label_set_attr(ipc_service_port_label_t port_splabel, mach_port_name_t name, mach_port_context_t context);

void
ipc_service_port_label_get_attr(ipc_service_port_label_t port_splabel, mach_port_name_t *name, mach_port_context_t *context);

#if CONFIG_SERVICE_PORT_INFO
void
ipc_service_port_label_get_info(ipc_service_port_label_t port_splabel, mach_service_port_info_t info);
#endif /* CONFIG_SERVICE_PORT_INFO */

#endif /* MACH_KERNEL_PRIVATE */
#endif /* _IPC_IPC_SERVICE_PORT_H_ */
