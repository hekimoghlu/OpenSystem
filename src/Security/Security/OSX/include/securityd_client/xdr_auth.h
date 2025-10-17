/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#include "sec_xdr.h"
#include <mach/message.h>
#include <Security/Authorization.h>

#ifdef __cplusplus
extern "C" {
#endif

bool_t xdr_AuthorizationItem(XDR *xdrs, AuthorizationItem *objp);
bool_t xdr_AuthorizationItemSet(XDR *xdrs, AuthorizationItemSet *objp);
bool_t xdr_AuthorizationItemSetPtr(XDR *xdrs, AuthorizationItemSet **objp);

bool_t copyin_AuthorizationItemSet(const AuthorizationItemSet *rights, void **copy, mach_msg_size_t *size);
bool_t copyout_AuthorizationItemSet(const void *copy, mach_msg_size_t size, AuthorizationItemSet **rights);

#ifdef __cplusplus
} // extern "C"
#endif
