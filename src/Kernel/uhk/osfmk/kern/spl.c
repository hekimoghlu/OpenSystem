/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 5, 2023.
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
#include <kern/thread.h>
#include <kern/spl.h>
#include <machine/machine_routines.h>

/*
 *  spl routines
 */

__private_extern__ spl_t
splhigh(
	void)
{
	return ml_set_interrupts_enabled(FALSE);
}

__private_extern__ spl_t
splsched(
	void)
{
	return ml_set_interrupts_enabled(FALSE);
}

__private_extern__ spl_t
splclock(
	void)
{
	return ml_set_interrupts_enabled(FALSE);
}

__private_extern__ void
spllo(
	void)
{
	(void)ml_set_interrupts_enabled(TRUE);
}

__private_extern__ void
splx(
	spl_t l)
{
	ml_set_interrupts_enabled((boolean_t) l);
}
