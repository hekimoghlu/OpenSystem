/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 1998-2011 The OpenLDAP Foundation.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */

#ifndef LDAP_RQ_H
#define LDAP_RQ_H 1

#include <ldap_cdefs.h>

LDAP_BEGIN_DECL

typedef struct re_s {
	struct timeval next_sched;
	struct timeval interval;
	LDAP_STAILQ_ENTRY(re_s) tnext; /* it includes running */
	LDAP_STAILQ_ENTRY(re_s) rnext;
	ldap_pvt_thread_start_t *routine;
	void *arg;
	char *tname;
	char *tspec;
} re_t;

typedef struct runqueue_s {
	LDAP_STAILQ_HEAD(l, re_s) task_list;
	LDAP_STAILQ_HEAD(rl, re_s) run_list;
	ldap_pvt_thread_mutex_t	rq_mutex;
} runqueue_t;

LDAP_F( struct re_s* )
ldap_pvt_runqueue_insert(
	struct runqueue_s* rq,
	time_t interval,
	ldap_pvt_thread_start_t* routine,
	void *arg,
	char *tname,
	char *tspec
);

LDAP_F( struct re_s* )
ldap_pvt_runqueue_find(
	struct runqueue_s* rq,
	ldap_pvt_thread_start_t* routine,
	void *arg
);

LDAP_F( void )
ldap_pvt_runqueue_remove(
	struct runqueue_s* rq,
	struct re_s* entry
);

LDAP_F( struct re_s* )
ldap_pvt_runqueue_next_sched(
	struct runqueue_s* rq,
	struct timeval* next_run
);

LDAP_F( void )
ldap_pvt_runqueue_runtask(
	struct runqueue_s* rq,
	struct re_s* entry
);

LDAP_F( void )
ldap_pvt_runqueue_stoptask(
	struct runqueue_s* rq,
	struct re_s* entry
);

LDAP_F( int )
ldap_pvt_runqueue_isrunning(
	struct runqueue_s* rq,
	struct re_s* entry
);

LDAP_F( void )
ldap_pvt_runqueue_resched(
	struct runqueue_s* rq,
	struct re_s* entry,
	int defer
);

LDAP_F( int )
ldap_pvt_runqueue_persistent_backload(
	struct runqueue_s* rq
);

LDAP_END_DECL

#endif
