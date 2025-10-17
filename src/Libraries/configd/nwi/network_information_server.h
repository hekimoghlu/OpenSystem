/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#ifndef _S_NETWORK_INFORMATION_SERVER_H
#define _S_NETWORK_INFORMATION_SERVER_H

#include <sys/cdefs.h>
#include <stdbool.h>
#include <mach/mach.h>
#include <CoreFoundation/CoreFoundation.h>
#include <network_information.h>

typedef void (^_nwi_sync_handler_t)(Boolean inSync);

__BEGIN_DECLS

void
load_NetworkInformation		(CFBundleRef		bundle,
				 _nwi_sync_handler_t	syncHandler);

void
_nwi_state_signature		(nwi_state_t		state,
				 unsigned char		*signature,
				 size_t			signature_len);

_Bool
_nwi_state_store		(nwi_state_t		state);

__END_DECLS

#endif	/* !_S_NETWORK_INFORMATION_SERVER_H */
