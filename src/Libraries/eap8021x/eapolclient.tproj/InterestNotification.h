/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
 * InterestNotification.h
 * - register for IOKit interest notification on a particular BSD name
 */

/* 
 * Modification History
 *
 * March 10, 2009	Dieter Siegmund (dieter@apple)
 * - created
 */

#ifndef _S_INTERESTNOTIFICATION_H
#define _S_INTERESTNOTIFICATION_H

struct InterestNotification;

typedef struct InterestNotification * InterestNotificationRef;

typedef void (*InterestNotificationCallbackRef)(InterestNotificationRef interest_p, const void * arg);


InterestNotificationRef
InterestNotificationCreate(const char * if_name, 
			   InterestNotificationCallbackRef callback,
			   const void * arg);
void
InterestNotificationRelease(InterestNotificationRef interest_p);

#endif /* _S_INTERESTNOTIFICATION_H */
