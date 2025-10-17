/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
/* 
 * Modification History
 *
 * November 8, 2001	Dieter Siegmund
 * - created
 */

#ifndef _S_ALERT_H
#define _S_ALERT_H

#include <sys/types.h>
#include <net/if_dl.h>

typedef void (AlertCallback)(void * arg1, void * arg2);

typedef struct Alert_s Alert;

Alert *
Alert_create(AlertCallback * func,
	     void * arg1, void * arg2, 
	     char * title, char * message);

void
Alert_free(Alert * * alert_p_p);

#endif _S_ALERT_H

