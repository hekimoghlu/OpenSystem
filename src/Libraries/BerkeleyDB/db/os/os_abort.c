/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
 * __os_abort --
 *
 * PUBLIC: void __os_abort __P((ENV *));
 */
void
__os_abort(env)
	ENV *env;
{
	__os_stack(env);		/* Try and get a stack trace. */

#ifdef HAVE_ABORT
	abort();			/* Try and drop core. */
	/* NOTREACHED */
#endif
#ifdef SIGABRT
	(void)raise(SIGABRT);		/* Try and drop core. */
#endif
	exit(1);			/* Quit anyway. */
	/* NOTREACHED */
}
