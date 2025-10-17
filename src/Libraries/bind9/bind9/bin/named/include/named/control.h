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
/* $Id: control.h,v 1.38 2012/01/31 23:47:31 tbox Exp $ */

#ifndef NAMED_CONTROL_H
#define NAMED_CONTROL_H 1

/*! \file
 * \brief
 * The name server command channel.
 */

#include <isccc/types.h>

#include <isccfg/aclconf.h>

#include <named/types.h>

#define NS_CONTROL_PORT			953

#define NS_COMMAND_STOP		"stop"
#define NS_COMMAND_HALT		"halt"
#define NS_COMMAND_RELOAD	"reload"
#define NS_COMMAND_RECONFIG	"reconfig"
#define NS_COMMAND_REFRESH	"refresh"
#define NS_COMMAND_RETRANSFER	"retransfer"
#define NS_COMMAND_DUMPSTATS	"stats"
#define NS_COMMAND_QUERYLOG	"querylog"
#define NS_COMMAND_DUMPDB	"dumpdb"
#define NS_COMMAND_SECROOTS	"secroots"
#define NS_COMMAND_TRACE	"trace"
#define NS_COMMAND_NOTRACE	"notrace"
#define NS_COMMAND_FLUSH	"flush"
#define NS_COMMAND_FLUSHNAME	"flushname"
#define NS_COMMAND_FLUSHTREE	"flushtree"
#define NS_COMMAND_STATUS	"status"
#define NS_COMMAND_TSIGLIST	"tsig-list"
#define NS_COMMAND_TSIGDELETE	"tsig-delete"
#define NS_COMMAND_FREEZE	"freeze"
#define NS_COMMAND_UNFREEZE	"unfreeze"
#define NS_COMMAND_THAW		"thaw"
#define NS_COMMAND_TIMERPOKE	"timerpoke"
#define NS_COMMAND_RECURSING	"recursing"
#define NS_COMMAND_NULL		"null"
#define NS_COMMAND_NOTIFY	"notify"
#define NS_COMMAND_VALIDATION	"validation"
#define NS_COMMAND_SCAN 	"scan"
#define NS_COMMAND_SIGN 	"sign"
#define NS_COMMAND_LOADKEYS 	"loadkeys"
#define NS_COMMAND_ADDZONE	"addzone"
#define NS_COMMAND_DELZONE	"delzone"
#define NS_COMMAND_SYNC		"sync"
#define NS_COMMAND_SIGNING	"signing"
#define NS_COMMAND_ZONESTATUS	"zonestatus"

isc_result_t
ns_controls_create(ns_server_t *server, ns_controls_t **ctrlsp);
/*%<
 * Create an initial, empty set of command channels for 'server'.
 */

void
ns_controls_destroy(ns_controls_t **ctrlsp);
/*%<
 * Destroy a set of command channels.
 *
 * Requires:
 *	Shutdown of the channels has completed.
 */

isc_result_t
ns_controls_configure(ns_controls_t *controls, const cfg_obj_t *config,
		      cfg_aclconfctx_t *aclconfctx);
/*%<
 * Configure zero or more command channels into 'controls'
 * as defined in the configuration parse tree 'config'.
 * The channels will evaluate ACLs in the context of
 * 'aclconfctx'.
 */

void
ns_controls_shutdown(ns_controls_t *controls);
/*%<
 * Initiate shutdown of all the command channels in 'controls'.
 */

isc_result_t
ns_control_docommand(isccc_sexpr_t *message, isc_buffer_t *text);

#endif /* NAMED_CONTROL_H */
