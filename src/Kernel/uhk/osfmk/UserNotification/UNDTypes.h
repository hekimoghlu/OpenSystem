/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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
#ifndef __USERNOTIFICATION_UNDTYPES_H
#define __USERNOTIFICATION_UNDTYPES_H

#include <sys/appleapiopts.h>

#ifdef __APPLE_API_PRIVATE

#include <mach/mach_types.h>

typedef char *UNDMessage;
typedef char *UNDLabel;
typedef char *UNDKey;
typedef char *UNDPath;

/*
 * serialized key's, list delimiters, ...
 *	(sent as out-of-line data in a message)
 */
typedef const char * xmlData_t;

#ifdef KERNEL_PRIVATE
#ifdef MACH_KERNEL_PRIVATE

/*
 * UNDReply definition - used to dispatch UserNotification
 * replies back to the in-kernel client.
 */
typedef struct UNDReply *UNDReplyRef;

#include <sys/cdefs.h>
__BEGIN_DECLS
extern UNDReplyRef convert_port_to_UNDReply(mach_port_t);
__END_DECLS

#else /* !MACH_KERNEL_PRIVATE */

typedef struct __UNDReply__ *UNDReplyRef;

#endif /* !MACH_KERNEL_PRIVATE */

#else /* ! KERNEL_PRIVATE */

typedef mach_port_t UNDReplyRef;

#endif /* ! KERNEL_PRIVATE */

#define UND_REPLY_NULL ((UNDReplyRef)0)
#define XML_DATA_NULL   ((xmlData_t)0)

#endif  /* __APPLE_API_PRIVATE */

#endif  /* __USERNOTIFICATION_UNDTPES_H */
