/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * November 9, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#ifndef _CACHE_H
#define _CACHE_H

#include <os/availability.h>
#include <TargetConditionals.h>
#include <sys/cdefs.h>

__BEGIN_DECLS

void	do_block		(int argc, char * const argv[]);

void	do_list			(int argc, char * const argv[]);
void	do_add			(int argc, char * const argv[]);
void	do_get			(int argc, char * const argv[]);
void	do_set			(int argc, char * const argv[]);
void	do_show			(int argc, char * const argv[]);
void	do_remove		(int argc, char * const argv[]);
void	do_notify		(int argc, char * const argv[]);

__END_DECLS

#endif	/* !_CACHE_H */
