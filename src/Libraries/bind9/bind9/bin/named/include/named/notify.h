/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
/* $Id: notify.h,v 1.16 2009/01/17 23:47:42 tbox Exp $ */

#ifndef NAMED_NOTIFY_H
#define NAMED_NOTIFY_H 1

#include <named/types.h>
#include <named/client.h>

/***
 ***	Module Info
 ***/

/*! \file
 * \brief
 *	RFC1996
 *	A Mechanism for Prompt Notification of Zone Changes (DNS NOTIFY)
 */

/***
 ***	Functions.
 ***/

void
ns_notify_start(ns_client_t *client);

/*%<
 *	Examines the incoming message to determine appropriate zone.
 *	Returns FORMERR if there is not exactly one question.
 *	Returns REFUSED if we do not serve the listed zone.
 *	Pass the message to the zone module for processing
 *	and returns the return status.
 *
 * Requires
 *\li	client to be valid.
 */

#endif /* NAMED_NOTIFY_H */

