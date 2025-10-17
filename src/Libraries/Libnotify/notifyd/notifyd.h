/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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
#ifndef _NOTIFY_DAEMON_H_
#define _NOTIFY_DAEMON_H_

#define DISPATCH_MACH_SPI 1

#include <libnotify.h>
#include <mach/mach.h>
#include <launch.h>
#include <dispatch/dispatch.h>
#include <dispatch/private.h>


#define STATUS_REQUEST_SHORT 0
#define STATUS_REQUEST_LONG 1

#define NOTIFY_STATE_ENTITLEMENT "com.apple.private.libnotify.statecapture"


struct global_s
{
	notify_state_t notify_state;
	dispatch_mach_t mach_notifs_channel;
	mach_port_t mach_notify_port;
	mach_port_t server_port;
	void **service_info_list;
	dispatch_workloop_t workloop;
	dispatch_mach_t mach_channel;
	dispatch_source_t sig_usr1_src;
	dispatch_source_t sig_usr2_src;
	dispatch_source_t sig_winch_src;
	dispatch_source_t stat_reset_src;
	dispatch_source_t memory_pressure_src;
	time_t last_reset_time;
	uint32_t nslots;
	uint32_t slot_id;
	uint32_t *shared_memory_base;
	uint32_t *shared_memory_refcount;
	uint32_t *last_shm_base;
	int log_cutoff;
	uint32_t log_default;
	uint32_t next_no_client_token;
	uint16_t service_info_count;
	char *log_path;
};

extern struct global_s global;

struct call_statistics_s
{
	uint64_t post;
	uint64_t post_no_op;
	uint64_t post_by_id;
	uint64_t post_by_name;
	uint64_t post_by_name_and_fetch_id;
	uint64_t reg;
	uint64_t reg_plain;
	uint64_t reg_check;
	uint64_t reg_signal;
	uint64_t reg_file;
	uint64_t reg_port;
	uint64_t reg_xpc_event;
	uint64_t reg_common;
	uint64_t cancel;
	uint64_t suspend;
	uint64_t resume;
	uint64_t suspend_pid;
	uint64_t resume_pid;
	uint64_t check;
	uint64_t get_state;
	uint64_t get_state_by_client;
	uint64_t get_state_by_id;
	uint64_t get_state_by_client_and_fetch_id;
	uint64_t set_state;
	uint64_t set_state_by_client;
	uint64_t set_state_by_id;
	uint64_t set_state_by_client_and_fetch_id;
	uint64_t set_owner;
	uint64_t set_access;
	uint64_t monitor_file;
	uint64_t service_path;
	uint64_t cleanup;
	uint64_t regenerate;
	uint64_t checkin;
};

extern struct call_statistics_s call_statistics;

extern void log_message(int priority, const char *str, ...) __printflike(2, 3);
extern uint32_t daemon_post(const char *name, uint32_t u, uint32_t g);
extern uint32_t daemon_post_nid(uint64_t nid, uint32_t u, uint32_t g);
extern void daemon_post_client(uint64_t cid);
extern void daemon_set_state(const char *name, uint64_t val);
extern void dump_status(uint32_t level, int fd);
extern bool has_entitlement(audit_token_t audit, const char *entitlement);
extern bool has_root_entitlement(audit_token_t audit);

void notifyd_matching_register(uint64_t event_token, xpc_object_t descriptor);
void notifyd_matching_unregister(uint64_t event_token);

dispatch_queue_t get_notifyd_workloop(void);

#endif /* _NOTIFY_DAEMON_H_ */
