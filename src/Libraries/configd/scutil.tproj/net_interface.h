/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 10, 2023.
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
 * August 5, 2004			Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _NET_INTERFACE_H
#define _NET_INTERFACE_H

#include <sys/cdefs.h>

__BEGIN_DECLS

CF_RETURNS_RETAINED
CFStringRef		_interface_description	(SCNetworkInterfaceRef interface);

SCNetworkInterfaceRef	_find_interface		(int argc, char * const argv[], int *nArgs);

void			create_interface	(int argc, char * const argv[]);
void			select_interface	(int argc, char * const argv[]);
void			set_interface		(int argc, char * const argv[]);
void			show_interface		(int argc, char * const argv[]);
void			show_interfaces		(int argc, char * const argv[]);

__END_DECLS

#endif /* !_NET_INTERFACE_H */
