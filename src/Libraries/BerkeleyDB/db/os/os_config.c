/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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
#include "db_config.h"

#include "db_int.h"

/*
 * __os_fs_notzero --
 *	Return 1 if allocated filesystem blocks are not zeroed.
 *
 * PUBLIC: int __os_fs_notzero __P((void));
 */
int
__os_fs_notzero()
{
	/* Most filesystems zero out implicitly created pages. */
	return (0);
}

/*
 * __os_support_direct_io --
 *	Return 1 if we support direct I/O.
 *
 * PUBLIC: int __os_support_direct_io __P((void));
 */
int
__os_support_direct_io()
{
	int ret;

	ret = 0;

#ifdef HAVE_O_DIRECT
	ret = 1;
#endif
#if defined(HAVE_DIRECTIO) && defined(DIRECTIO_ON)
	ret = 1;
#endif
	return (ret);
}

/*
 * __os_support_db_register --
 *	Return 1 if the system supports DB_REGISTER.
 *
 * PUBLIC: int __os_support_db_register __P((void));
 */
int
__os_support_db_register()
{
	return (1);
}

/*
 * __os_support_replication --
 *	Return 1 if the system supports replication.
 *
 * PUBLIC: int __os_support_replication __P((void));
 */
int
__os_support_replication()
{
	return (1);
}
