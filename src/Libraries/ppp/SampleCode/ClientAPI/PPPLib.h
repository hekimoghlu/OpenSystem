/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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
 * Feb 10, 2001			Allan Nathanson <ajn@apple.com>
 * - cleanup API
 *
 * Feb 2000			Christophe Allie <callie@apple.com>
 * - initial revision
 */

#ifndef _PPPLIB_H
#define _PPPLIB_H

#include <sys/cdefs.h>
#include <CoreFoundation/CoreFoundation.h>
#include "ppp_msg.h"

__BEGIN_DECLS

int		PPPInit			(int			*ref);

int		PPPDispose		(int			ref);

int		PPPConnect		(int			ref,
					 u_int8_t 		*serviceid);

int		PPPDisconnect		(int			ref,
					 u_int8_t 		*serviceid);

int		PPPGetOption		(int			ref,
					 u_int8_t 		*serviceid,
					 u_int32_t		option,
					 void			**data,
					 u_int32_t		*dataLen);

int		PPPSetOption		(int			ref,
					 u_int8_t 		*serviceid,
					 u_int32_t		option,
					 void			*data,
					 u_int32_t		dataLen);

int		PPPStatus		(int			ref,
					 u_int8_t 		*serviceid,
					 struct ppp_status	**stat);

int		PPPEnableEvents		(int			ref,
					 u_int8_t 		*serviceid,
					 u_char			enable);

__END_DECLS

#endif	/* _PPPLIB_H */
