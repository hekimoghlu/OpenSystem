/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
/*!
 *       @header init.h
 *       This header defines an API to register a function that will be called when
 *       the network stack is being initialized. This gives a kernel extensions an
 *       opportunity to install filters before sockets are created and network
 *       operations occur.
 */
#ifndef _NET_INIT_H_
#define _NET_INIT_H_
#include <sys/kernel_types.h>

/*!
 *       @typedef net_init_func_ptr
 *       @discussion net_init_func_ptr will be called once the networking stack
 *               initialized and before network operations occur.
 */
typedef void    (*net_init_func_ptr)(void);

/*!
 *       @function net_init_add
 *       @discussion Add a function to be called during network initialization. Your
 *               kext must not unload until the function you  register is called if
 *               net_init_add returns success.
 *       @param init_func A pointer to a function to be called when the stack is
 *               initialized.
 *       @result	EINVAL - the init_func value was NULL.
 *                       EALREADY - the network has already been initialized
 *                       ENOMEM - there was not enough memory to perform this operation
 *                       0 - success
 */
errno_t net_init_add(net_init_func_ptr  init_func);

#ifdef BSD_KERNEL_PRIVATE
/* net_init_run is called from bsd_init */
extern void net_init_run(void);
#endif /* BSD_KERNEL_PRIVATE */

#endif /* _NET_INIT_H_ */
