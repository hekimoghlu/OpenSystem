/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
#ifndef __IPSEC_MANAGER__
#define __IPSEC_MANAGER__

#include <sys/kern_event.h>

enum {
    IPSEC_NOT_ASSERTED = 0,
    IPSEC_ASSERTED_IDLE,
    IPSEC_ASSERTED_INITIALIZE,
    IPSEC_ASSERTED_CONTACT,
    IPSEC_ASSERTED_PHASE1,
    IPSEC_ASSERTED_PHASE2
};

#define IPSEC_ASSERT(ipsec, x) do {											\
										if (ipsec.phase == IPSEC_RUNNING) {	\
											ipsec.asserted = x;					\
										}											\
									} while (0)
#define IPSEC_UNASSERT(ipsec) (ipsec.asserted = IPSEC_NOT_ASSERTED)

#define IPSEC_ASSERT_IDLE(ipsec) IPSEC_ASSERT(ipsec, IPSEC_ASSERTED_IDLE)
#define IPSEC_ASSERT_INITIALIZE(ipsec) IPSEC_ASSERT(ipsec, IPSEC_ASSERTED_INITIALIZE)
#define IPSEC_ASSERT_CONTACT(ipsec) IPSEC_ASSERT(ipsec, IPSEC_ASSERTED_CONTACT)
#define IPSEC_ASSERT_PHASE1(ipsec) IPSEC_ASSERT(ipsec, IPSEC_ASSERTED_PHASE1)
#define IPSEC_ASSERT_PHASE2(ipsec) IPSEC_ASSERT(ipsec, IPSEC_ASSERTED_PHASE2)

#define IPSEC_IS_ASSERTED(ipsec, x) (ipsec.phase == IPSEC_RUNNING && ipsec.asserted == x)
#define IPSEC_IS_NOT_ASSERTED(ipsec) (ipsec.asserted == 0)
#define IPSEC_IS_ASSERTED_IDLE(ipsec) IPSEC_IS_ASSERTED(ipsec, IPSEC_ASSERTED_IDLE)
#define IPSEC_IS_ASSERTED_INITIALIZE(ipsec) IPSEC_IS_ASSERTED(ipsec, IPSEC_ASSERTED_INITIALIZE)
#define IPSEC_IS_ASSERTED_CONTACT(ipsec) IPSEC_IS_ASSERTED(ipsec, IPSEC_ASSERTED_CONTACT)
#define IPSEC_IS_ASSERTED_PHASE1(ipsec) IPSEC_IS_ASSERTED(ipsec, IPSEC_ASSERTED_PHASE1)
#define IPSEC_IS_ASSERTED_PHASE2(ipsec) IPSEC_IS_ASSERTED(ipsec, IPSEC_ASSERTED_PHASE2)
#define IPSEC_IS_ASSERTED_ANY(ipsec) (ipsec.phase == IPSEC_RUNNING && ipsec.asserted)

/* try to handle as many types of dns delimiters as possible */
#define GET_SPLITDNS_DELIM(data, delim) do {	\
		if (strstr(data, ",")) {				\
			delim = ",";						\
		} else if (strstr(data, ";")) {			\
			delim = ";";						\
		} else if (strstr(data, "\n")) {		\
			delim = "\n";						\
		} else if (strstr(data, "\r")) {		\
			delim = "\r";						\
		} else if (strstr(data, " ")) {			\
			delim = " ";						\
		} else {								\
			delim = "\0";						\
		}										\
	} while(0)

u_int16_t ipsec_subtype(CFStringRef subtypeRef);

int ipsec_new_service(struct service *serv);
int ipsec_dispose_service(struct service *serv);
int ipsec_setup_service(struct service *serv);
void ipsec_set_initial_values(struct service *serv, CFDictionaryRef initialValues);

int ipsec_start(struct service *serv, CFDictionaryRef options, uid_t uid, gid_t gid, mach_port_t bootstrap, u_int8_t onTraffic, u_int8_t onDemand);
int ipsec_stop(struct service *serv, int signal);
int ipsec_getstatus(struct service *serv);
int ipsec_copyextendedstatus(struct service *serv, CFDictionaryRef *statusdict);
int ipsec_copystatistics(struct service *serv, CFDictionaryRef *statsdict);
int ipsec_getconnectdata(struct service *serv, CFDictionaryRef *options, int all);

int ipsec_install(struct service *serv);
int ipsec_uninstall(struct service *serv);

int ipsec_can_sleep(struct service *serv);
int ipsec_will_sleep(struct service *serv, int checking);
void ipsec_wake_up(struct service *serv);
void ipsec_device_lock(struct service *serv);
void ipsec_device_unlock(struct service *serv);
void ipsec_log_out(struct service *serv);
void ipsec_log_in(struct service *serv);
void ipsec_log_switch(struct service *serv);
void ipsec_ipv4_state_changed(struct service *serv);
void ipsec_user_notification_callback(struct service *serv, CFUserNotificationRef userNotification, CFOptionFlags responseFlags);
int ipsec_ondemand_add_service_data(struct service *serv, CFMutableDictionaryRef ondemand_dict);
void ipsec_cellular_event(struct service *serv, int event);
void ipsec_network_event(struct service *serv, struct kern_event_msg *ev_msg);

int ipsec_init_things(void);

#endif
