/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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
 * October 4, 2004			Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _NET_SET_H
#define _NET_SET_H

#include <sys/cdefs.h>

__BEGIN_DECLS

void	create_set		(int argc, char * const argv[]);
void	remove_set		(int argc, char * const argv[]);
void	select_set		(int argc, char * const argv[]);
void	set_set			(int argc, char * const argv[]);
void	show_set		(int argc, char * const argv[]);
void	show_sets		(int argc, char * const argv[]);

__END_DECLS

#endif /* !_NET_SET_H */
